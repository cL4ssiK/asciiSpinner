from PIL import Image
import trimesh
import numpy as np
import sys


#----------------------2d image into ascii art-------------------------------------------------------

def muunnakuva(img, characters="@%#*+=-:. "):
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

#----------------------3d model into ascii art--------------------------------------------------------


# Transform .obj into mesh object.
def obj_to_mesh(objfile):
    mesh = trimesh.load(objfile)
    #jos on useita objecteja tehdään scene.
    if isinstance(mesh, list):
        mesh = trimesh.util.concatenate(mesh)
    elif isinstance(mesh, trimesh.Scene):
        meshes = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(meshes)
    return mesh


# Reduce quality of model.
def simplify(mesh, target_faces=5000):
    #simplified.export("cat_simplified.obj")
    return mesh.simplify_quadric_decimation(face_count=target_faces)


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


# Rotate model n degrees to any direction.
def rotate(vert, angle):
    angle = np.asarray(angle)
    angle = np.deg2rad(angle)
    m_x = rotation_matrix_x(angle[0])
    m_y = rotation_matrix_y(angle[1])
    m_z = rotation_matrix_z(angle[2])
    rotationmatrix = m_x @ m_y @ m_z
    return vert @ rotationmatrix.T


# Calculate normalized normal vectors for every face.
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


# Calculate how bright each face is to the camera.
def face_brightnesses(normals, light_vector=[0, 0, 1]):
    normals = np.asarray(normals)
    light_vector = np.asarray(light_vector)

    brightness_values = np.maximum(0, np.dot(normals, light_vector))
    return brightness_values


# Determines what faces will be shown on screen. TODO:tätä on muokattava
def determine_printed_vertices(face_projection, screen_height=100, screen_width=120):
    depth_buffer = np.full((screen_height, screen_width), None, dtype=object)
    for v in face_projection:
        int_x = int(screen_width/2) + int(round(v[0][0]))
        int_y = int(screen_height/2) + int(round(v[0][1]))
        if depth_buffer[int_x][int_y] == None or (depth_buffer[int_x][int_y])[0] < v[0][2]:
            depth_buffer[int_x][int_y] = (v[0][2], v[1])
    
    return depth_buffer


def determine_ascii_char(brightnesses):
    characters = np.array(list(" .:-=+*#%@"))
    vali = 255 / len(characters)
    indexes = np.round(np.round(brightnesses*255) / vali).astype(int)
    indexes = np.where(indexes > 0, indexes - 1, 0)
    asciiarr = characters[indexes]
    return asciiarr


# Calculates center point for each face.
def face_centers(faces, projections):
    faces = np.asarray(faces)
    projections = np.asarray(projections)

    v0 = projections[faces[:, 0]]
    v1 = projections[faces[:, 1]]
    v2 = projections[faces[:, 2]]

    centers = (v0 + v1 + v2)/3
    return centers


# Creates tuples from face center vectors and face textures (characters).
def join_character_to_coordinates(face_centers, brightness):
    face_centers = np.asarray(face_centers)
    brightness = np.asarray(brightness)
    characters = determine_ascii_char(brightness)

    character_and_center = []
    for v, c in zip(face_centers, characters):
        character_and_center.append((v, c))
    return character_and_center


# Replaces none values from depthbuffer with ' '
def remove_Nones(db):
    for row in range(len(db)):
        for col in range(len(db[row])):
            if db[row][col] == None:
                db[row][col] = (0, ' ')


#TODO: koita clearausta siirtämällä cursori alkuun tai ansi escape koodi
#TODO: Kokeile jossain vaihees laskea facejen keskipisteet ja normaalivektorit vaan kerran, ja sit suorittaa kääntäminen niille. 
#TODO: Tee daemon thread jonka avulla ensin voidaan kääntää paikallaan oleva kuva oikein päin ja sit alottaa mallin pyörittäminen.
#TODO: Tee myös mahdollisuus siirtää mallia edestakaisin nuolista. esim: näppäin M asettaa liikutustilan jolloin nuolet liikuttaa kuvaa,
#       R asettaa rotate tilaan, jolloin käännetään nuolen suuntaan kuvaa. D asettaa sitten display tilaan, jolloin malli alkaa pyöriä.
def main():

    MAXIUM_FACES = 10000
    THRESHOLD_PERCENTAGE = 0.8

    arguments = sys.argv[1:]
    model_file = arguments[0]

    mesh = obj_to_mesh(model_file)

    previous = len(mesh.faces)
    while len(mesh.faces) > MAXIUM_FACES:
        mesh = simplify(mesh)
        if(len(mesh.faces) > previous*THRESHOLD_PERCENTAGE and len(mesh.faces) > MAXIUM_FACES):
            print("Model is too detailed for ascii graphics!")
            return
        previous = len(mesh.faces)

    mesh.vertices = rotate(mesh.vertices, [0, -90, 0])

    angle=[0,0,0]
    while True:
        angle = [5, 0, 0]
        #angle = [5, 1, 2]
        mesh.vertices = rotate(mesh.vertices, angle)

        normals = face_normal_vectors(mesh.vertices, mesh.faces)

        bri = face_brightnesses(normals)

        chc = face_centers(mesh.faces, mesh.vertices)

        jj = join_character_to_coordinates(chc, bri)

        depthbuffer = determine_printed_vertices(jj)

        remove_Nones(depthbuffer)

        print("\033c", end="")

        for rivi in depthbuffer:
            print("".join(str(cell[1]) for cell in rivi))

if __name__ == "__main__":
    main()

#img = Image.open("donitsi.jpg").convert("L")
#muunnos = muunnakuva(img)
#for rivi in muunnos:
#    print("".join(rivi))