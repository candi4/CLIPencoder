from PIL import Image
import PIL

a = Image.open('Usage/images/maze2.png')
print(type(a))
print(isinstance(a , Image.Image))