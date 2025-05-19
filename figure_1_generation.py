from pnpdm.models.sd_wrappers.daps_sd_wrapper import DapsSDWrapper
import torch
from torchvision import transforms
from pnpdm.data import get_dataset, get_dataloader
from torchvision.utils import save_image

def norm_image_01(x):
    return (x * 0.5 + 0.5).clip(0, 1)


def save_grid(images, target="image.png", nrow=10, normalize=True):
    # Save images.
    if normalize:
        images = norm_image_01(images)
    save_image(images, target, nrow=nrow)


transform = transforms.Compose([transforms.Normalize((0.5), (0.5))])
dataset = get_dataset(
    name="images_with_prompts",
    root="full_final_images",
    prompts_file="full_final_images/prompts.json",
    transform=transform,
)
prompts = [dataset.get_prompt(i) for i in range(len(dataset))]
num_test_images = len(dataset)
dataloader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

DEVICE = "cuda:0"
model = DapsSDWrapper(device=DEVICE)

for im, prompt in zip(dataloader, prompts):
    if prompt != "two scoops of ice cream with a cannoli":
        continue

    im = im.to(DEVICE).half()
    latent = model.encode_image(im)

    sigma = model.get_sigma(33)
    noisy = latent + torch.randn_like(latent) * sigma

    model.set_prompt("a plate of ice cream")
    clean1 = model.sample(noisy, starting_sigma=sigma)
    model.set_prompt("a plate of macarons")
    clean2 = model.sample(noisy, starting_sigma=sigma)
    model.set_prompt("a plate of cupcakes")
    clean3 = model.sample(noisy, starting_sigma=sigma)

    for i,j in enumerate([noisy, clean1, clean2, clean3]):
        save_grid(
            model.decode_image(j),
            target=f"im{i}.png",
        )
