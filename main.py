from train import train_model
from visualize import plot_two_models


def train_menu(model_type):
    print("\nВыберите режим обучения:\n"
          "1. Дообучение (Adam)\n"
          "2. Дообучение (NovoGrad)\n"
          "3. Обучение с нуля (Adam)\n"
          "4. Назад\n")
    choice = input()

    match choice:
        case "1":
            train_model(pretrained=True, model_path=model_type[0], lr=5e-5, epochs=30, use_novograd=False)
        case "2":
            train_model(pretrained=True, model_path=model_type[1], lr=5e-5, epochs=30, use_novograd=True)
        case "3":
            train_model(pretrained=False, model_path=model_type[2], lr=5e-5, epochs=30, use_novograd=False)
        case "4":
            return
        case _:
            print("Некорректный выбор")


if __name__ == "__main__":

    model_path = [
        "models/mobilenet_finetune_adam.pth",
        "models/mobilenet_finetune_novograd.pth",
        "models/mobilenet_scratch_adam.pth"
    ]

    while True:
        print("\nГлавное меню:\n"
              "1. Обучить модели\n"
              "2. Сравнить Adam и NovoGrad\n"
              "3. Сравнить обучение с нуля и дообучение\n"
              "4. Выход\n")

        x = input()

        match x:
            case "1":
                train_menu(model_path)
            case "2":
                plot_two_models(
                    "CSV/mobilenet_finetune_adam_training_history.csv",
                    "CSV/mobilenet_finetune_novograd_training_history.csv"
                )
            case "3":
                plot_two_models(
                    "CSV/mobilenet_finetune_adam_training_history.csv",
                    "CSV/mobilenet_scratch_adam_training_history.csv"
                )
            case "4":
                break
            case _:
                print("Некорректный выбор")
