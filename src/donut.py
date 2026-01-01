import trimesh
import numpy as np
import sys
from scipy.spatial import ConvexHull
import math
import time

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
def determine_printed_vertices(face_projection, screen_height=110, screen_width=120):
    depth_buffer = np.full((screen_height, screen_width), None, dtype=object)
    for v in face_projection:
        int_y = int(screen_width/2) + int(round(v[0][1]))
        int_x = int(screen_height/2) + int(round(v[0][0]))
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


'''
Moves the vertices for x, y, z amount. 
We need to expand the matrix into fourth dimension temporarily to acheive the move with multiplication.
'''
def move(vertices, x, y, z):
    ones = np.ones((vertices.shape[0], 1))
    vertices_h = np.hstack([vertices, ones])
    T = np.array([
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, z],
    [0, 0, 0, 1]
    ])
    moved_vertices_h = vertices_h @ T.T  # (N, 4)
    moved_vertices = moved_vertices_h[:, :3]
    return moved_vertices


def size_of_model(vertices):
    # Build convex hull
    hull = ConvexHull(vertices)
    hull_points = vertices[hull.vertices]

    # Brute-force only on hull vertices
    max_dist = 0
    for i in range(len(hull_points)):
        dists = np.linalg.norm(hull_points[i] - hull_points, axis=1)
        j = np.argmax(dists)
        if dists[j] > max_dist:
            max_dist = dists[j]

    return max_dist

'''
Calculates center point of the model from maxium and minium coordinates for each axis.
Bounds box around the model, calculates space diagonal vector and halves it.
'''
def model_center_point(opposite_corners):
    return (opposite_corners[0] + opposite_corners[1]) / 2


def model_furthest_points(vertices):
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    return np.vstack([mins, maxs])

'''
Now only for x axis direction spinning.
TODO: make this universal.
'''
def determine_buffer_size(model_dimensions_xyz):
    #TODO: Keksi miten tää tehään. Jotenkin täytyy selvittää missä suunnassa tarvitaan mitenkin paljon tilaa.
    #       koska se minkä akselien suhteen pyöritetään vaikuttaa suurimpaan leveyteen ja korkeuteen, 
    #       täytyy pyörimiskulma huomioida laskussa. 
    # Periaatteessa muodostetaan ympyrät pyörimisakselien ympärille. esim jos pyörii y,x ympäri suhteessa
    #   5:1 pyörähtää kappale alkuasennossaan kokonaan kummankin akselin ympäri. Täten tarvitaan tilaa korkeus-
    #   jokin ratkaisu on ottaa maksimi niiden akselien mukaisista dimensioista, joiden suuntaan ei pyöritä. muuten vain sen akselin mitta
    width = max(int(math.ceil(model_dimensions_xyz[1])), int(math.ceil(model_dimensions_xyz[2])))
    height = int(math.ceil(model_dimensions_xyz[0]))
    return width, height

#TODO: koita clearausta siirtämällä cursori alkuun tai ansi escape koodi
#TODO: Kokeile jossain vaihees laskea facejen keskipisteet ja normaalivektorit vaan kerran, ja sit suorittaa kääntäminen niille. 
#TODO: Tee daemon thread jonka avulla ensin voidaan kääntää paikallaan oleva kuva oikein päin ja sit alottaa mallin pyörittäminen.
#TODO: Tee myös mahdollisuus siirtää mallia edestakaisin nuolista. esim: näppäin M asettaa liikutustilan jolloin nuolet liikuttaa kuvaa,
#       R asettaa rotate tilaan, jolloin käännetään nuolen suuntaan kuvaa. D asettaa sitten display tilaan, jolloin malli alkaa pyöriä.
def main():

    MAXIUM_FACES = 10000
    THRESHOLD_PERCENTAGE = 0.8

    ROTATION_SPEED = 90  # degrees per second
    MAX_FRAMES = 30
    frame_time = 1/MAX_FRAMES

    BASE_ANGLE = np.array([1, 0, 0])

    #colorama.init()

    arguments = sys.argv[1:]
    #model_file = "cat_simplified.obj"#arguments[0]
    model_file = "../assets/cat.obj"#arguments[0]

    mesh = obj_to_mesh(model_file)

    previous = len(mesh.faces)
    while len(mesh.faces) > MAXIUM_FACES:
        mesh = simplify(mesh)
        if(len(mesh.faces) > previous*THRESHOLD_PERCENTAGE and len(mesh.faces) > MAXIUM_FACES):
            print("Model is too detailed for ascii graphics!")
            return
        previous = len(mesh.faces)


    fps = model_furthest_points(mesh.vertices)
    center_point  = model_center_point(fps)

    # Rotate to right orientation.
    mesh.vertices = rotate(mesh.vertices, [0, -90, 0])

    fps = model_furthest_points(mesh.vertices)
    center_point  = model_center_point(fps)
    dimensions_xyz = np.abs(fps[0] - fps[1])
    buffer_w, buffer_h = determine_buffer_size(dimensions_xyz)

    mesh.vertices = move(mesh.vertices, -center_point[0], -center_point[1], -center_point[2])
    
    last_time = time.time()

    #TODO: tee funktio joka muodostaa aina framen. sitten mieti pitäisikö bufferin koko laskea face centereistä.

    while True:
        now = time.time()      
        dt = now - last_time
        last_time = now

        angle = BASE_ANGLE * ROTATION_SPEED * dt

        mesh.vertices = rotate(mesh.vertices, angle)

        normals = face_normal_vectors(mesh.vertices, mesh.faces)

        bri = face_brightnesses(normals)

        chc = face_centers(mesh.faces, mesh.vertices)

        jj = join_character_to_coordinates(chc, bri)

        depthbuffer = determine_printed_vertices(jj, buffer_h+4, buffer_w+4)

        remove_Nones(depthbuffer)

        print("\033c", end="")
        # this could work on some terminals.
        #n = len(depthbuffer)
        #sys.stdout.write(f"\033[{n+1}A")
        #sys.stdout.write("\033[H")
        #sys.stdout.flush()

        for rivi in depthbuffer:
            print("".join(str(cell[1]) for cell in rivi))
        
        elapsed = time.time() - now
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()

#img = Image.open("donitsi.jpg").convert("L")
#muunnos = muunnakuva(img)
#for rivi in muunnos:
#    print("".join(rivi))