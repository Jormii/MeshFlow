from PIL import Image

from complex_mesh import ComplexMesh
from obj_parser import parse_obj


def main():
    obj_mesh_path = "./simple.obj"
    obj_parse = parse_obj(obj_mesh_path)
    meshes = obj_parse.meshes()

    for mesh in meshes:
        complex_mesh = ComplexMesh(mesh)
        complex_mesh.plot()

        textures = complex_mesh.calculate_flow()
        for i, (texture, negative_texture) in enumerate(textures):
            img_tex = Image.fromarray(texture)
            img_tex.save("./Results/{}_flow_{}.png".format(mesh.name, i))

            img_neg = Image.fromarray(negative_texture)
            img_neg.save("./Results/{}_flow_neg_{}.png".format(mesh.name, i))


if __name__ == "__main__":
    main()
