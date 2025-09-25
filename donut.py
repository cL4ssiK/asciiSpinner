from PIL import Image
import trimesh
import numpy as np

#characters = "$@%&#*/\\|()\{\}[]?-_+~<>!;:,\"^`'. "
#kaikki = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1\{\}[]?-_+~<>i!lI;:,\"^`'. "


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


# muuta obj mesh objektiksi
def obj_to_mesh(objfile):
    mesh = trimesh.load(objfile)
    #jos on useita objecteja tehdään scene.
    if isinstance(mesh, list):
        mesh = trimesh.util.concatenate(mesh)
    elif isinstance(mesh, trimesh.Scene):
        meshes = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(meshes)
    return mesh


def rotation_matrix_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])


def rotation_matrix_y(theta):
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def rotation_matrix_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])


# käännä mallia x astetta
def rotate(vert, angle, axis='z'):
    rotationmatrix=None
    angle = np.deg2rad(angle)
    if axis == 'x':
        rotationmatrix = rotation_matrix_x(angle)
    elif axis == 'y':
        rotationmatrix = rotation_matrix_y(angle)
    elif axis == 'z':
        rotationmatrix = rotation_matrix_z(angle)
    
    if rotationmatrix is None:
        return vert
    return vert @ rotationmatrix.T


# laske normaalivektorit faceille.
def face_normal_vectors(vertices, faces):
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    normalvectors = np.cross(e1, e2)
    normalized_length = np.linalg.norm(normalvectors, axis=1, keepdims=True)
    normalvectors /= normalized_length
    return normalvectors


# lasketaan miten kirkas kukin kohta pinnasta on.
def face_brightnesses(normals, light_vector=[0, 0, 1]):
    normals = np.asarray(normals)
    light_vector = np.asarray(light_vector)

    brightness_values = np.maximum(0, np.dot(normals, light_vector))
    return brightness_values


# määrittää mikä printataan ja minne. TODO:tätä on muokattava
def determine_printed_vertices(face_projection, screen_height=100, screen_width=120):
    depth_buffer = np.full((screen_height, screen_width), None, dtype=object)
    for v in face_projection:
        int_x = int(screen_width/2) + int(round(v[0][0]))
        int_y = int(screen_height/2) + int(round(v[0][1]))
        if depth_buffer[int_x][int_y] == None or (depth_buffer[int_x][int_y])[0] < v[0][2]:
            depth_buffer[int_x][int_y] = (v[0][2], v[1])
    
    return depth_buffer


def determine_ascii_char(brightness):
    return muunna_yksi_merkki(int(round(brightness*255)), " .:-=+*#%@")


def determine_ascii_char2(brightnesses):
    characters = np.array(list(" .:-=+*#%@"))
    vali = 255 / len(characters)
    indexes = np.round(np.round(brightnesses*255) / vali).astype(int)
    indexes = np.where(indexes > 0, indexes - 1, 0)
    asciiarr = characters[indexes]
    return asciiarr


# laskee keskipisteen kullekin facelle. pitänee jättää syvyys pois???
def face_centers(faces, projections):
    faces = np.asarray(faces)
    projections = np.asarray(projections)

    v0 = projections[faces[:, 0]]
    v1 = projections[faces[:, 1]]
    v2 = projections[faces[:, 2]]

    centers = (v0 + v1 + v2)/3
    return centers


# tehdään tupleja, jossa toisena osana paikkavektori, ja toisena osana piirrettävä merkki.
def join_character_to_coordinates(face_centers, brightness):
    face_centers = np.asarray(face_centers)
    brightness = np.asarray(brightness)
    characters = determine_ascii_char2(brightness)

    character_and_center = []
    for v, c in zip(face_centers, characters):
        character_and_center.append((v, c))
    return character_and_center


def remove_Nones(db):
    for row in range(len(db)):
        for col in range(len(db[row])):
            if db[row][col] == None:
                db[row][col] = (0, ' ')

#TODO: koita clearausta siirtämällä cursori alkuun tai ansi escape koodi
#TODO: joku ongelma tossa kääntämisessä. selvitä.

#TODO: Kokeile jossain vaihees laskea facejen keskipisteet ja normaalivektorit vaan kerran, ja sit suorittaa kääntäminen niille. 

#img = Image.open("donitsi.jpg").convert("L")
#muunnos = muunnakuva(img)
#for rivi in muunnos:
#    print("".join(rivi))

mesh = obj_to_mesh("cat_simplified.obj")
mesh.vertices = rotate(mesh.vertices, -90, axis='y')
angle=0
while True:
    angle = 5
    mesh.vertices = rotate(mesh.vertices, angle, axis='x')
    #print("vertices: ", len(mesh.vertices))
    #print("faces: ", len(mesh.faces))
    normals = face_normal_vectors(mesh.vertices, mesh.faces)
    #print("normals: ", len(normals))
    bri = face_brightnesses(normals)
    #print("brightnesses: ", len(bri))
    #chc = character_coordinates(mesh.faces, mesh.vertices)
    chc = face_centers(mesh.faces, mesh.vertices)
    #print("proj. face centers: ", len(chc))
    jj = join_character_to_coordinates(chc, bri)
    depthbuffer = determine_printed_vertices(jj)
    remove_Nones(depthbuffer)
    #os.system('cls' if os.name == 'nt' else 'clear')
    print("\033c", end="")
    for rivi in depthbuffer:
        #print("".join(str(cell[1]) if cell is not None else " " for cell in rivi))
        print("".join(str(cell[1]) for cell in rivi))


#rotated = rotate(mesh.vertices, angle)

#vertice formaatti: [x, y, z]



#simplified = mesh.simplify_quadric_decimation(face_count=5000)
#print("Simplified vertices:", len(simplified.vertices))
#print("Simplified faces:", len(simplified.faces))
#
#simplified.export("cat_simplified.obj")

'''
eli. koska jokainen face sisältää indeksit alkuperäiseen vertice taulukkoon minkä vektorien kautta face piirretään,
pysyvät nämä pisteet samoina myös muunnoksessa. Eli siis uuden taulukon samoista indekseistä löytyy uudet vektorit,
joiden läpi face nyt piirretään.
Nyt koska haluamme piirtää nämä facet, otamme aina facea vastaavat verticet uudesta taulukosta, ja laskemme facen 
keskipisteen. Haluamme piirtää merkin tähän pisteeseen. Kuitenkin päällekäin tulevia pisteitä syntyy useita, joten 
huomioitava on vain lähimpänä oleva piste.
Täytynee myös hieman skaalata keskipisteiden koordinaatteja, sillä muuten kuvasta tulisi mega ahdas.

kysymysmerkkejä:
    - muodostuuko ongelma, sillä facet eivät välttämättä ole kokonaan päällekäin, vaan tulee selvittää ovatko osittain 
        päällekäin.
    - merkki on iso, täten pieni pinta saattaa suurentua liikaa 
    - miten päällekäisyys ratkaistaan ylipäänsä.
'''