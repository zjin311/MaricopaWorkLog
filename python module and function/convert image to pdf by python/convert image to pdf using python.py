#%%
from PIL import Image

image1 = Image.open('C:/Users/Mr.Goldss/Desktop/covid -19 employee payroll certification.jpg')
im1 = image1.convert('RGB')
im1.save('C:/Users/Mr.Goldss/Desktop/covid -19 employee payroll certification.pdf')

