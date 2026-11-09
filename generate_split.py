import json
import random
import os


def generate_split(total_subjects=30, train_size=20, val_size=5, test_size=5):
    if train_size + val_size + test_size != total_subjects:
        print(
            f"Error: Suma de sujetos ({train_size+val_size+test_size}) no coincide con el total ({total_subjects})"
        )
        return

    subjects = list(range(1, total_subjects + 1))
    random.shuffle(subjects)

    train_subjects = subjects[:train_size]
    val_subjects = subjects[train_size : train_size + val_size]
    test_subjects = subjects[train_size + val_size :]

    split = {
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
    }

    with open("data_split.json", "w") as f:
        json.dump(split, f, indent=4)

    print(f"Split generado en data_split.json:")
    print(f"Train: {train_subjects}")
    print(f"Val:   {val_subjects}")
    print(f"Test:  {test_subjects}")


if __name__ == "__main__":
    # Puedes cambiar la semilla para reproducibilidad
    random.seed(32345)
    generate_split()
