from PIL import Image

characters = "$@%&#*/\\|()\{\}[]?-_+~<>!;:,\"^`'. "
kaikki = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1\{\}[]?-_+~<>i!lI;:,\"^`'. "
        
def muunnakuva(data, characters="@%#*+=-:. "):
    w, h = img.size

    pixels = list(img.getdata())

    # reshape into 2D grid
    pixel_grid = [pixels[i * w:(i + 1) * w] for i in range(h)]

    asciimerkit = [] 
    for row in pixel_grid:
        asciirow = []
        for pixel in row:
            m = muunna_yksi_merkki(pixel, characters)
            asciirow.append(m)
            asciirow.append(m)
        asciimerkit.append(asciirow)
    return asciimerkit

def muunna_yksi_merkki(pixelvalue, merkit):
    vali = 255 / len(merkit)
    index = int(round(pixelvalue/vali))
    if (index > 0): 
        index -= 1
    return merkit[index]


img = Image.open("Shrek.png").convert("L")
muunnos = muunnakuva(img)
for rivi in muunnos:
    print("".join(rivi))