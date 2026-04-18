"""Quick validation: train all 8 models and save the best one."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

from src.preprocessing import load_data, full_preprocessing, prepare_train_test
from src.models import get_all_models
from src.evaluate import compare_models, save_best_model

df = load_data("data/train.csv")
df = full_preprocessing(df)
X_train, X_test, y_train, y_test, scaler, features = prepare_train_test(df)

models = get_all_models()
results_df, trained_models = compare_models(models, X_train, X_test, y_train, y_test)

best_name = results_df.index[0]
best_model = trained_models[best_name]
save_best_model(best_model, best_name, scaler, features, "models/best_model.pkl")

acc = results_df.loc[best_name, "Test Accuracy"]
print(f"\n[BEST] {best_name} -> Accuracy: {acc:.4f}")
