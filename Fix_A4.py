#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inkscape extension to correct perspective of raster images.
Tested with Inkscape version 1.3.2

Author: Shrinivas Kulkarni (khemadeva@gmail.com)

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


Modification AlphaJet64
"""


import base64
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen


import cv2
import inkex
from inkex.utils import debug
import numpy as np
from inkex import Image

import os
from tkinter import Tk, filedialog, messagebox 


def get_full_href(href: str, svg_file_path: str) -> str:
    """
    Get the full href for a given relative or absolute path.

    Args:
        href (str): The original href.
        svg_file_path (str): The path to the SVG file.

    Returns:
        str: The full href.
    """
    parsed_href = urlparse(href)

    if parsed_href.scheme == "file":
        path = Path(parsed_href.path)
        if path.is_absolute():
            return href
        else:
            new_path = Path(svg_file_path) / path
            new_href = urlunparse(parsed_href._replace(path=str(new_path)))
            return "file://" + str(new_href)
    elif parsed_href.scheme:
        return href
    else:
        path = Path(href)
        if path.is_absolute():
            return "file://" + str(path)
        else:
            new_path = Path(svg_file_path) / path
            return "file://" + str(new_path)


def get_cv_image(element: inkex.Image, svg_file_path: str) -> np.ndarray:
    """
    Get the OpenCV image from an embedded or linked image of Inkscape image element.

    Args:
        element (inkex.Image): The Inkscape image element.
        svg_file_path (str): The path to the SVG file.

    Returns:
        np.ndarray: The OpenCV image.
    """
    href = element.get("{http://www.w3.org/1999/xlink}href") or ""

    if href.startswith("data:image"):
        _, encoded = href.split(",", 1)
        img_data = base64.b64decode(encoded)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        cvImage = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
##        debug(f"Je suis une image intégrée") 
    else:
        href = get_full_href(href, svg_file_path)
        resp = urlopen(href)
        img_array = np.asarray(bytearray(resp.read()), dtype="uint8")
        cvImage = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cvImage


def put_cv_image(cvImage: np.ndarray, element: inkex.Image) -> None:
    """
    Update the Inkscape image element with the new OpenCV image.

    Args:
        cvImage (np.ndarray): The OpenCV image.
        element (inkex.Image): The Inkscape image element.
    """
    href = element.get("{http://www.w3.org/1999/xlink}href") or ""

    if not href.startswith("data:image"):
        inkex.errormsg(
          "Une nouvelle image 'redressement_A4' a été créée dans le dessin."
        )
    _, buffer = cv2.imencode(".jpg", cvImage, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{img_base64}"
    element.set("{http://www.w3.org/1999/xlink}href", data_uri)
    # else:
    #     image_path = get_full_href(href, svg_file_path)
    #     cv2.imwrite(image_path.replace("file://", ""), cvImage)


def get_elem_transform(
    elem: inkex.BaseElement, scale_x: float = 1, scale_y: float = 1
) -> inkex.Transform:
    """
    Get the transform for an Inkscape Image element.

    Args:
        elem (inkex.BaseElement): The Inkscape element.
        scale_x (float, optional): The x-scale factor. Defaults to 1.
        scale_y (float, optional): The y-scale factor. Defaults to 1.

    Returns:
        inkex.Transform: The resulting transform.
    """
    transform = inkex.Transform(elem.transform)
    img_translation = [float(elem.get("x") or "0"), float(elem.get("y") or "0")]
    transform = transform.add_translate(*img_translation)
    transform.add_scale(scale_x, scale_y)
    return transform

def apply_perspective(
    cvImage: np.ndarray, corners: List[inkex.Vector2d], width: int, height: int
) -> Tuple[np.ndarray, int, int]:
    """
    Apply perspective transformation to the image with specified output width and height.

    Args:
        cvImage (np.ndarray): The input OpenCV image.
        corners (List[inkex.Vector2d]): The four corners of the perspective rectangle.
        width (int): The target width of the output image.
        height (int): The target height of the output image.

    Returns:
        Tuple[np.ndarray, int, int]: The transformed image, width, and height.
    """

    def sorted_corners(corners: np.ndarray) -> np.ndarray:
        centroid = np.mean(corners, axis=0)
        # Calculate angles from centroid
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        # Sort indices based on angles
        sorted_indices = np.argsort(angles)
        # Arrange corners counter-clockwise starting from top-right
        return corners[sorted_indices]

    corners_array = np.array(corners, dtype="float32")
    corners_array = sorted_corners(corners_array)

    # Define target corners based on width and height (will be paper dimensions)
    target_corners = np.array(
        [
            [0, 0],
            [int(width) , 0],
            [int(width) , int(height)],
            [0, int(height) ],
        ],
        dtype="float32",
    )

    # Get the perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(corners_array, target_corners)
    new_image = cv2.warpPerspective(cvImage, matrix, (int(width), int(height)))

    return new_image, int(width), int(height)


# Sauvegarde l'image


def get_default_image_directory() -> Path:
    """
    Détermine le répertoire par défaut des images selon l'OS.
    Retourne le chemin du dossier d'images de l'utilisateur.
    """
    home_dir = Path.home()
    if os.name == 'posix':  # Linux ou macOS
        image_dir = home_dir / "Images"
    elif os.name == 'nt':  # Windows
        image_dir = home_dir / "Pictures"
    else:
        image_dir = home_dir  # Si le dossier d'images par défaut n'est pas trouvé, on utilise le répertoire utilisateur
    
    # Crée le dossier s'il n'existe pas
    image_dir.mkdir(exist_ok=True)
    return image_dir

def save_cv_image(cvImage: np.ndarray, echelle: float = 1.0, default_filename="redresse_A4") -> bool:
    """
    Sauvegarde une image OpenCV dans le dossier par défaut des images du système.

    Args:
        cvImage (np.ndarray): L'image OpenCV à sauvegarder.
        default_filename (str): Nom de fichier par défaut proposé pour la sauvegarde.
    """
    # Récupère le répertoire d'images par défaut
    image_dir = get_default_image_directory()

    # Prépare le chemin de fichier complet pour la sauvegarde
    default_path = image_dir / f"{default_filename}.png"

    # Initialiser Tkinter pour le dialogue
    root = Tk()
    root.withdraw()  # Masquer la fenêtre principale Tkinter

    # Demander confirmation pour la sauvegarde
    save_image = messagebox.askyesno("Sauvegarde de l'image", "Voulez-vous sauvegarder l'image ?")

    if save_image:
        # Ouvrir un dialogue de sauvegarde dans le dossier d'images par défaut
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_filename,
            initialdir=image_dir,
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
            # Redimensionner l'image en fonction de l'échelle
        if echelle != 1.0:
            largeur = int(cvImage.shape[1] * echelle)
            hauteur = int(cvImage.shape[0] * echelle)
            dimensions = (largeur, hauteur)
            cvImage = cv2.resize(cvImage, dimensions, interpolation=cv2.INTER_LINEAR)

 
        # Si un fichier est sélectionné, sauvegarder l'image
        if file_path:
            try:
                if file_path.lower().endswith('.png'):
                    cv2.imwrite(file_path, cvImage, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                elif file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                    cv2.imwrite(file_path, cvImage, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                else:
                    messagebox.showerror("Erreur", "Format de fichier non supporté.")
                    return
                messagebox.showinfo("Succès", f"Image sauvegardée sous : {file_path}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Échec de la sauvegarde de l'image : {str(e)}")
        else:
            print("Aucun fichier sélectionné pour la sauvegarde.")
    else:
        print("Sauvegarde annulée.")



class RasterPerspectiveEffect(inkex.Effect):
    """
    Inkscape extension to correct perspective of raster images.
    """

    def __init__(self):
        super().__init__()

    def effect(self):

        
        # Unit du viewport
        
        try:
            unit_to_vp = self.svg.unit_to_viewport
        except AttributeError:
            unit_to_vp = self.svg.uutounit

        try:
            vp_to_unit = self.svg.viewport_to_unit
        except AttributeError:
            vp_to_unit = self.svg.unittouu

        # Récupérer les dimensions de la feuille
        paper_width_str = self.document.getroot().get('width')
        paper_height_str = self.document.getroot().get('height')

        # Convertir les dimensions en nombres
        paper_width = float(paper_width_str[:-2]) if paper_width_str.endswith('mm') else float(paper_width_str)
        paper_height = float(paper_height_str[:-2]) if paper_height_str.endswith('mm') else float(paper_height_str)

        # Déterminer les unités
        drawing_units = 'mm' if paper_width_str.endswith('mm') else 'px'

        
        img_elem = path_elem = None
        for elem in self.svg.selection:
            if isinstance(elem, inkex.Image):
                if not img_elem:
                    img_elem = elem
            elif elem is not None and len(list(elem.path.end_points)) >= 4:
                path_elem = elem
            if path_elem is not None and img_elem is not None:
                break

        if img_elem is None or path_elem is None:
            inkex.errormsg("Veuillez d'abord sélectionner une image et un quadrilatère (fermé ou non)")
            return

        file_path = self.svg_path()
        if not file_path:
            return

        cvImage = get_cv_image(img_elem, file_path)

        scale_y, scale_x = [
            d1 / d0
            for (d0, d1) in zip(cvImage.shape[:2], [img_elem.height, img_elem.width])
        ]
        
##        debug(f"Echelle x : {scale_x} , Echelle y : {scale_y} ") 
        img_transform = get_elem_transform(img_elem, scale_x, scale_y)

        corners = [
            path_elem.composed_transform().apply_to_point(pt)
            for pt in list(path_elem.path.end_points)
        ][:4]
        local_corners = [(-img_transform).apply_to_point(pt) for pt in corners]

        cvImage, width, height = apply_perspective(cvImage, local_corners,int(paper_width), int(paper_height))
        
        img_elem_copie = Image(
            id="redressement_A4",  # ID de la nouvelle image
        )
        self.svg.get_current_layer().add(img_elem_copie)

        img_elem_copie.set("x", 0)
        img_elem_copie.set("y", 0)

        scale = unit_to_vp(paper_width)/paper_width
        debug(f"echelle : {scale}")

        if save_cv_image(cvImage,scale):
            print("Image sauvegardée avec succès.")
        else:
            print("La sauvegarde de l'image a échoué ou a été annulée.")
        put_cv_image(cvImage, img_elem_copie)           

if __name__ == "__main__":
    RasterPerspectiveEffect().run()
