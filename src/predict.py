"""
Inference script for predicting real estate prices in Tashkent.
Usage:
    python -m src.predict --rooms 3 --size 75 --level 4 --max_levels 9 --district "Чиланзарский"
"""

import os
import pickle
import argparse
import pandas as pd


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")


def load_model_and_features():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Trained model not found in {MODEL_DIR}. Please run 'python -m src.train' first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_columns = pickle.load(f)
    return model, feature_columns


def predict_price(rooms: int, size: float, level: int, max_levels: int, district: str = "", location: str = "") -> float:
    """Estimates the property price given apartment features."""
    model, feature_columns = load_model_and_features()

    # Create empty feature row initialized to zeros
    input_dict = {col: 0 for col in feature_columns}

    # Set numerical fields
    if "rooms" in input_dict:
        input_dict["rooms"] = rooms
    if "size" in input_dict:
        input_dict["size"] = size
    if "level" in input_dict:
        input_dict["level"] = level
    if "max_levels" in input_dict:
        input_dict["max_levels"] = max_levels

    # Set one-hot encoded district and location if present
    district_col = f"district_{district}"
    if district_col in input_dict:
        input_dict[district_col] = 1

    location_col = f"location_{location}"
    if location_col in input_dict:
        input_dict[location_col] = 1

    input_df = pd.DataFrame([input_dict], columns=feature_columns)
    predicted = model.predict(input_df)[0]
    return predicted


def main():
    parser = argparse.ArgumentParser(description="Predict Tashkent Real Estate Price")
    parser.add_argument("--rooms", type=int, required=True, help="Number of rooms")
    parser.add_argument("--size", type=float, required=True, help="Apartment area in square meters")
    parser.add_argument("--level", type=int, required=True, help="Floor level of the apartment")
    parser.add_argument("--max_levels", type=int, required=True, help="Total number of floors in the building")
    parser.add_argument("--district", type=str, default="Чиланзарский", help="District name (e.g., Юнусабадский, Чиланзарский, Мирзо-Улугбекский)")
    parser.add_argument("--location", type=str, default="", help="Exact location string if known")

    args = parser.parse_args()

    price = predict_price(
        rooms=args.rooms,
        size=args.size,
        level=args.level,
        max_levels=args.max_levels,
        district=args.district,
        location=args.location,
    )

    print("\n==========================================")
    print("🏠 Tashkent House Price Prediction Result")
    print("==========================================")
    print(f"  Rooms:       {args.rooms}")
    print(f"  Size:        {args.size} m²")
    print(f"  Floor:       {args.level} / {args.max_levels}")
    print(f"  District:    {args.district}")
    print("------------------------------------------")
    print(f"  Estimated Price: ${price:,.2f} USD")
    print("==========================================\n")


if __name__ == "__main__":
    main()
