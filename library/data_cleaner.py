import pandas as pd
import numpy as np

class DataCleaner:
    """
    Utilities for preprocessing datasets, computing class balance,
    and auditing training/validation splits for leakage.
    Built with NumPy and Pandas.
    """
    @staticmethod
    def calculate_imbalance_ratio(df, target_col):
        """
        Calculates the class imbalance ratio (majority count / minority count).
        """
        counts = df[target_col].value_counts()
        if len(counts) < 2:
            return 1.0
        majority = counts.iloc[0]
        minority = counts.iloc[-1]
        return majority / (minority or 1)

    @staticmethod
    def impute_missing(train_df, val_df, strategy="median"):
        """
        Imputes missing values.
        Important: Fits the statistics on the train set only, and transforms both train and val.
        """
        train_clean = train_df.copy()
        val_clean = val_df.copy()

        for col in train_clean.columns:
            if train_clean[col].isnull().sum() > 0 or val_clean[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(train_clean[col]):
                    if strategy == "mean":
                        fill_val = train_clean[col].mean()
                    else:
                        fill_val = train_clean[col].median()
                else:
                    fill_val = train_clean[col].mode().iloc[0] if not train_clean[col].mode().empty else "missing"
                
                train_clean[col] = train_clean[col].fillna(fill_val)
                val_clean[col] = val_clean[col].fillna(fill_val)

        return train_clean, val_clean

    @staticmethod
    def scale_features(train_df, val_df, numeric_cols, method="standard"):
        """
        Applies scaling to numeric columns.
        Fits parameters (mean/std or min/max) on the train set only to prevent leakage.
        """
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()

        for col in numeric_cols:
            if method == "minmax":
                col_min = train_df[col].min()
                col_max = train_df[col].max()
                denom = (col_max - col_min) or 1.0
                
                train_scaled[col] = (train_df[col] - col_min) / denom
                val_scaled[col] = (val_df[col] - col_min) / denom
            else: # Standard scaling
                col_mean = train_df[col].mean()
                col_std = train_df[col].std()
                denom = col_std if (col_std and not np.isnan(col_std)) else 1.0
                
                train_scaled[col] = (train_df[col] - col_mean) / denom
                val_scaled[col] = (val_df[col] - col_mean) / denom

        return train_scaled, val_scaled

    @staticmethod
    def run_leakage_audit(train_df, val_df, target_col, entity_col=None, temporal_col=None):
        """
        Audits train/validation splits for leakage.
        Returns:
            dict: { passed: bool, flags: list, details: str }
        """
        passed = True
        flags = []
        details = []

        # 1. Split Integrity Check (Overlap checking)
        # Check for exact duplicates across splits (ignoring target)
        train_features = train_df.drop(columns=[target_col], errors='ignore')
        val_features = val_df.drop(columns=[target_col], errors='ignore')

        # Convert rows to tuple hashes for quick set comparison
        train_hashes = set(train_features.apply(lambda row: hash(tuple(row)), axis=1))
        val_hashes = val_features.apply(lambda row: hash(tuple(row)), axis=1)
        
        overlap_count = sum(h in train_hashes for h in val_hashes)
        overlap_pct = (overlap_count / len(val_df)) * 100 if len(val_df) > 0 else 0
        
        if overlap_pct > 0.5:
            passed = False
            flags.append("SPLIT-INTEGRITY-VIOLATION")
            details.append(f"Overlap detected: {overlap_pct:.2f}% of validation rows exist in training set.")

        # 2. Group Bleed Check
        if entity_col and entity_col in train_df.columns and entity_col in val_df.columns:
            train_entities = set(train_df[entity_col].dropna().unique())
            val_entities = set(val_df[entity_col].dropna().unique())
            bleed_entities = train_entities.intersection(val_entities)
            
            if len(bleed_entities) > 0:
                passed = False
                flags.append("GROUP-BLEED")
                details.append(f"Group bleed: {len(bleed_entities)} entity IDs overlap across train/val. Use GroupKFold.")

        # 3. Target Proxies Check
        # Inspect features with extremely high correlation with target
        # For numeric target: compute Pearson correlation. For classification: compute Cramer's V / proxy check.
        if pd.api.types.is_numeric_dtype(train_df[target_col]):
            corr_matrix = train_df.corr(numeric_only=True)
            if target_col in corr_matrix.columns:
                correlations = corr_matrix[target_col].abs().sort_values(ascending=False)
                # drop the target itself
                correlations = correlations.drop(index=[target_col], errors='ignore')
                
                leaky_features = correlations[correlations > 0.85]
                if len(leaky_features) > 0:
                    passed = False
                    flags.append("TARGET-PROXY-LEAK")
                    details.append(f"Leaky target proxies: Features {list(leaky_features.index)} explain > 85% of target variance.")

        # 4. Temporal Bleed Check
        if temporal_col and temporal_col in train_df.columns and temporal_col in val_df.columns:
            # For temporal split, every timestamp in val should be >= max timestamp in train
            max_train_time = train_df[temporal_col].max()
            min_val_time = val_df[temporal_col].min()
            
            if min_val_time < max_train_time:
                passed = False
                flags.append("TEMPORAL-BLEED")
                details.append(f"Temporal bleed: Val starts before train ends ({min_val_time} < {max_train_time}).")

        return {
            "passed": passed,
            "flags": flags,
            "details": "; ".join(details) if details else "Leakage audit passed cleanly."
        }
