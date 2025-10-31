from data.data_preprocessor import load_gait_frames
from models.gaitPCA import PCA, Recognizer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from PIL import Image
import numpy as np
import pandas as pd
import os
from datetime import datetime


def main():
    dataset_paths = {
        "Binary": "gait-model\data\Binary",
        "Multiclass4": "gait-model\data\Multiclass4",
        "Multiclass6": "gait-model\data\Multiclass6"
    }

    target_size = (64, 64)
    random_state = 42
    component_range = list(range(5, 26))

    results = []
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"pca_cv_log_{timestamp}.csv")

    for dataset_name, path in dataset_paths.items():
        print(f"\n=== Dataset: {dataset_name} ===")
        df = load_gait_frames(path, load_images=True)

        images = [img.resize(target_size, Image.BILINEAR) for img in df['image'].values]
        X = np.stack([np.array(img) for img in images])
        X = X.reshape(X.shape[0], -1)
        y = df['label'].values

        print("Total samples:", len(df))
        print("Class distribution:\n", pd.Series(y).value_counts())

        for num_components in component_range:
            print(f"\n>> Components: {num_components}")

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            fold_metrics = []

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                print(f"\n--- Fold {fold + 1} ---")

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                recognizer = Recognizer(num_components=num_components, method='svm')
                recognizer.train(X_train, y_train)
                metrics = recognizer.evaluate(X_test, y_test)

                # Clean and store metrics
                fold_metrics.append({
                    "dataset": dataset_name,
                    "num_components": num_components,
                    "fold": fold + 1,
                    "accuracy": metrics["accuracy"],
                    "rejection_rate": metrics["rejection_rate"],
                    "num_correct": metrics["num_correct"],
                    "num_unknown": metrics["num_unknown"],
                    "num_recognized": metrics["num_recognized"],
                    "total": metrics["total"],
                    "confusion_matrix": metrics["confusion_matrix"].tolist()
                })

                print(metrics)



            avg = {
                "dataset": dataset_name,
                "num_components": num_components,
                "fold": "avg",
                "accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
                "rejection_rate": np.mean([m["rejection_rate"] for m in fold_metrics]),
                "num_correct": np.mean([m["num_correct"] for m in fold_metrics]),
                "num_unknown": np.mean([m["num_unknown"] for m in fold_metrics]),
                "num_recognized": np.mean([m["num_recognized"] for m in fold_metrics]),
                "total": np.mean([m["total"] for m in fold_metrics]),
                "confusion_matrix": "avg"
            }

            results.extend(fold_metrics)
            results.append(avg)

    # Save all results
    df_results = pd.DataFrame(results)
    df_results.to_csv(log_path, index=False)
    print(f"\n✅ All results logged to: {log_path}")


if __name__ == "__main__":
    main()
