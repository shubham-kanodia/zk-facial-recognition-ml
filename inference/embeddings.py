import os
import torch

from facenet_pytorch import MTCNN, InceptionResnetV1


class Generator:
    def __init__(self):
        self.workers = 0 if os.name == 'nt' else 4
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))

        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )

        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def generate(self, image):
        image = image.convert("RGB")
        img_cropped = self.mtcnn(image)
        img_embedding = self.resnet(img_cropped.unsqueeze(0)).detach().numpy()

        return list(img_embedding.reshape(512))

