with open("eval_results.csv", "a") as f:
    f.write(f'{"model_path"},{"scale"},{"dataset"},{"max_psnr"},{"max_ssim"},{"max_mse"},{"mean_psnr"},{"mean_ssim"},{"mean_mse"}\n')
