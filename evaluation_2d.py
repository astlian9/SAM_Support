from train_2d import main as run_2d
import func_2d.function as function


def main():
    run_2d(evaluate_only=True)


def evaluate_sam(args, val_loader, support_loader, epoch, net, clean_dir=True):
    return function.validation_sam(
        args, val_loader, support_loader, epoch, net, clean_dir
    )


if __name__ == '__main__':
    main()
