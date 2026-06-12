import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


class ColabRunner:
    """
    Drives Google Colab through Playwright so training code runs on a Colab GPU
    without manual copy-paste.

    Lifecycle:
      1. login()    -- one-time, headed. Opens a visible Chromium window on the
                       Colab page and blocks until the researcher logs in to
                       Google. Auth cookies persist in PROFILE_DIR.
      2. connect()  -- headless. Relaunches Chromium with the saved profile and
                       opens (or creates) a notebook. From here the operator
                       controls Colab as if executing locally.
      3. run_cell() -- inserts code into a new cell, executes it, polls the
                       output until the run sentinel or a HARD-STOP marker
                       appears, and returns the scraped text.

    Caveats:
      - Colab's DOM drifts. Cell creation and execution use keyboard shortcuts
        (stable); output scraping uses the selectors in _scrape_output --
        update them if Colab changes markup.
      - _scrape_output concatenates all sandboxed output frames on the page.
        The operator's run-ID gate (rules.md section 2.0) is what guarantees
        the wrong cell's output is never triaged.
      - Google sometimes refuses logins in automated browsers. The launch
        flags below reduce that; if refused, complete the login once in the
        headed window from login() -- the persisted profile carries auth.
    """

    PROFILE_DIR = str(Path.home() / ".ml-experiment-pilot" / "chromium-profile")
    COLAB_URL = "https://colab.research.google.com"
    NEW_NOTEBOOK_URL = "https://colab.research.google.com/#create=true"

    _LAUNCH_KWARGS = dict(
        ignore_default_args=["--enable-automation"],
        args=["--disable-blink-features=AutomationControlled"],
    )

    def __init__(self):
        self._pw = None
        self._ctx = None
        self.page = None

    # -- session lifecycle --------------------------------------------------

    def login(self, timeout_s=600):
        """One-time headed login. Blocks until Google login is detected, then closes the window."""
        with sync_playwright() as pw:
            ctx = pw.chromium.launch_persistent_context(
                self.PROFILE_DIR, headless=False, **self._LAUNCH_KWARGS
            )
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(self.COLAB_URL)
            print("A Chromium window is open on Google Colab.")
            print("Log in to your Google account there. The window closes by itself once login is detected.")
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                if self._signed_in(page):
                    print("Login detected. Auth saved to the local browser profile.")
                    print("All further Colab control is headless -- no more windows will open.")
                    ctx.close()
                    return True
                time.sleep(2)
            ctx.close()
            raise TimeoutError(f"No Google login detected within {timeout_s}s.")

    def connect(self, notebook_url=None):
        """
        Relaunch headless with the saved profile and open a notebook.
        Returns the notebook URL -- persist it in setup/environment.md as
        colab_notebook_url and pass it back in to reuse the same notebook.
        """
        self._pw = sync_playwright().start()
        self._ctx = self._pw.chromium.launch_persistent_context(
            self.PROFILE_DIR, headless=True, **self._LAUNCH_KWARGS
        )
        self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        self.page.goto(notebook_url or self.NEW_NOTEBOOK_URL, wait_until="load")
        if not self._signed_in(self.page):
            self.close()
            raise PermissionError(
                "Google login expired. Re-run: python library/colab_runner.py --login"
            )
        self.page.wait_for_selector("div.cell", timeout=60_000)
        return self.page.url

    def close(self):
        if self._ctx:
            self._ctx.close()
            self._ctx = None
        if self._pw:
            self._pw.stop()
            self._pw = None
        self.page = None

    @staticmethod
    def _signed_in(page):
        try:
            if "accounts.google" in page.url:
                return False
            return page.locator("text=Sign in").count() == 0
        except Exception:
            return False

    # -- execution ------------------------------------------------------------

    def ensure_gpu(self):
        """
        Best-effort: Runtime > Change runtime type > T4 GPU > Save.
        Always verify afterwards with a probe cell: torch.cuda.is_available().
        """
        try:
            self.page.get_by_text("Runtime", exact=True).first.click()
            self.page.get_by_text("Change runtime type").first.click()
            self.page.get_by_text("T4 GPU").first.click()
            self.page.get_by_role("button", name="Save").click()
            return True
        except Exception:
            return False

    def interrupt_cell(self):
        """
        Stop the currently executing cell -- the colab-mode equivalent of
        killing a run. Tries the keyboard chord (Ctrl+M I) first, then the
        Runtime menu. Returns True if the interrupt was issued; verify it
        landed by scraping for "KeyboardInterrupt" in the cell output.
        """
        try:
            self.page.locator("div.cell").last.click()
            self.page.keyboard.press("Escape")
            self.page.keyboard.press("Control+m")
            self.page.keyboard.press("i")
            return True
        except Exception:
            pass
        try:
            self.page.get_by_text("Runtime", exact=True).first.click()
            self.page.get_by_text("Interrupt execution").first.click()
            return True
        except Exception:
            return False

    def run_cell(self, code, sentinel, hard_stop_marker="HARD-STOP",
                 timeout_s=21600, poll_s=15, on_output=None):
        """
        Append a new cell containing `code`, execute it, and poll the scraped
        output until `sentinel` (e.g. "RUN-COMPLETE run-gbt-07") or
        `hard_stop_marker` appears. Returns the full scraped output text.

        on_output(text) is called with the current scrape on every poll --
        use it to relay checkpoint lines to the local dashboard.
        Raises TimeoutError if neither marker appears within timeout_s, and
        ConnectionError if the page or runtime goes away mid-run.
        """
        page = self.page
        cells = page.locator("div.cell")
        cells.last.click()
        page.keyboard.press("Escape")        # leave edit mode
        page.keyboard.press("Control+m")     # Colab chord prefix...
        page.keyboard.press("b")             # ...new cell below
        cells.last.click()                   # focus the new cell's editor
        page.keyboard.insert_text(code)
        page.keyboard.press("Control+Enter")
        self._dismiss_dialogs()

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                output = self._scrape_output()
            except Exception as exc:
                raise ConnectionError(f"Lost the Colab page mid-run: {exc}") from exc
            if on_output and output:
                on_output(output)
            if hard_stop_marker in output or sentinel in output:
                return output
            time.sleep(poll_s)
        raise TimeoutError(
            f"Neither '{sentinel}' nor '{hard_stop_marker}' appeared within {timeout_s}s."
        )

    def _dismiss_dialogs(self):
        for label in ("Run anyway", "OK"):
            try:
                self.page.get_by_role("button", name=label).click(timeout=1500)
            except Exception:
                pass

    def _scrape_output(self):
        """Concatenate the last cell's output area and every sandboxed output frame."""
        parts = []
        out = self.page.locator("div.cell").last.locator(".output")
        if out.count():
            try:
                parts.append(out.inner_text(timeout=5000))
            except Exception:
                pass
        for frame in self.page.frames:
            if "outputframe" in (frame.url or ""):
                try:
                    parts.append(frame.locator("body").inner_text(timeout=2000))
                except Exception:
                    pass
        return "\n".join(p for p in parts if p)


if __name__ == "__main__":
    if "--login" in sys.argv:
        ColabRunner().login()
    else:
        print("Usage: python colab_runner.py --login   (one-time headed Google login)")
        print("Programmatic use: ColabRunner().connect(...) then run_cell(...) -- see class docstring.")
