import joblib

model = joblib.load("ecommerce_return_model.pkl")

print("\nMODEL FEATURES:")
print(list(model.feature_names_in_))

print("\nPIPELINE STEPS:")
print(list(model.named_steps.keys()))

preprocessor = model.named_steps["preprocessor"]

print("\nPREPROCESSING FEATURES:")
for name, transformer, columns in preprocessor.transformers_:
    print(name, ":", list(columns))