import os

from train_3d import main as run_3d
import func_2d.function_3d as function


def main():
    run_3d(evaluate_only=True)


def evaluate_sam(args, val_loader, support_loader, epoch, net, clean_dir=True):
    return function.validation_sam(
        args, val_loader, support_loader, epoch, net, clean_dir,
        export_dir=os.path.join(args.path_helper['sample_path'], 'Volumes'),
    )


if __name__ == '__main__':
    main()
