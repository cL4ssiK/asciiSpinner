from PIL import Image
import trimesh
import numpy as np
import os

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
    if axis == 'x':
        rotationmatrix = rotation_matrix_x(angle)
    elif axis == 'y':
        rotationmatrix = rotation_matrix_y(angle)
    elif axis == 'z':
        rotationmatrix = rotation_matrix_z(angle)
    
    if rotationmatrix is None:
        return vert
    return vert @ rotationmatrix.T


#3d malli -> 2d kuva + syvyys TODO: muokkaa siten että syvyys ei kuulu vektoriin suoraan.
def projection_3D_to_2D(vertices):
    projection = []
    for v in vertices:
        projection.append(np.array([v[0], v[1], v[2]]))
    return projection


# laske normaalivektorit faceille.
def face_normal_vectors(vertices, faces):
    normalvectors = []
    for f in faces:
        vectors = [np.array(vertices[f[0]]), 
                   np.array(vertices[f[1]]), 
                   np.array(vertices[f[2]])]
        
        edges = [vectors[1]-vectors[0], vectors[2]-vectors[0]]
        normalvector = np.cross(edges[0], edges[1])
        normalized_for_direction = normalvector/np.linalg.norm(normalvector)
        normalvectors.append(normalized_for_direction)
    return normalvectors


# lasketaan miten kirkas kukin kohta pinnasta on.
def face_brightnesses(normals, light_vector=[0, 0, 1]):
    face_brightness = []
    for v in normals:
        face_brightness.append(max(0, np.dot(v, light_vector)))
    return face_brightness


# määrittää mikä printataan ja minne. TODO:tätä on muokattava
def determine_printed_vertices(face_projection, screen_height=100, screen_width=180):
    depth_buffer = np.full((screen_height, screen_width), None, dtype=object)
    for v in face_projection:
        int_x = int(screen_width/2) + int(round(v[0][0]))
        int_y = int(screen_height/2) + int(round(v[0][1]))
        if depth_buffer[int_x][int_y] == None or (depth_buffer[int_x][int_y])[0] < v[0][2]:
            depth_buffer[int_x][int_y] = (v[0][2], v[1])
    
    return depth_buffer


def determine_ascii_char(brightness):
    return muunna_yksi_merkki(int(round(brightness*255)), " .:-=+*#%@")


# laskee keskipisteen kullekin facelle. pitänee jättää syvyys pois???
def character_coordinates(faces, projections):
    char_pos = []
    for f in faces:
        center = (projections[f[0]] + projections[f[1]] + projections[f[2]])/3
        char_pos.append(center)
    return char_pos


# tehdään tupleja, jossa toisena osana paikkavektori, ja toisena osana piirrettävä merkki.
def join_character_to_coordinates(face_centers, brightness):
    tuplet = []
    for v, b in zip(face_centers, brightness):
        tuplet.append((v, determine_ascii_char(b)))
    return tuplet


def remove_Nones(db):
    for row in range(len(db)):
        for col in range(len(db[row])):
            if db[row][col] == None:
                db[row][col] = (0, ' ')

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
#img = Image.open("donitsi.jpg").convert("L")
#muunnos = muunnakuva(img)
#for rivi in muunnos:
#    print("".join(rivi))
mesh = obj_to_mesh("cat_simplified.obj")
mesh.vertices = rotate(mesh.vertices, -90, axis='y')
angle=0
while True:
    angle = angle + 10
    mesh.vertices = rotate(mesh.vertices, angle, axis='x')
    #print("vertices: ", len(mesh.vertices))
    #print("faces: ", len(mesh.faces))
    normals = face_normal_vectors(mesh.vertices, mesh.faces)
    #print("normals: ", len(normals))
    bri = np.array(face_brightnesses(normals))
    #print("brightnesses: ", len(bri))
    proj = projection_3D_to_2D(mesh.vertices)
    #print("proj. vertices: ", len(proj))
    chc = character_coordinates(mesh.faces, proj)
    #print("proj. face centers: ", len(chc))
    jj = join_character_to_coordinates(chc, bri)
    depthbuffer = determine_printed_vertices(jj)
    remove_Nones(depthbuffer)
    os.system('cls' if os.name == 'nt' else 'clear')
    for rivi in depthbuffer:
        print("".join(str(cell[1]) if cell is not None else " " for cell in rivi))


#rotated = rotate(mesh.vertices, angle)

#vertice formaatti: [x, y, z]



#simplified = mesh.simplify_quadric_decimation(face_count=5000)
#print("Simplified vertices:", len(simplified.vertices))
#print("Simplified faces:", len(simplified.faces))
#
#simplified.export("cat_simplified.obj")