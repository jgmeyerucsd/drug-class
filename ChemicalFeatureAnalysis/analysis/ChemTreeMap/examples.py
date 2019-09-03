#! /usr/bin/env python
#
# Copyright (C) 2016 Jing Lu <ajingnk@gmail.com>
# License: Apache

# -*- coding: utf-8 -*-

# pylint: disable=too-few-public-methods
import os

import treebuild
import shutil
from distutils.dir_util import copy_tree
from treebuild import DEFAULT_ACTIVITY_TYPES, DEFAULT_PROPERTY_TYPES, DEFAULT_FINGERPRINT_TYPES, DEFAULT_EXTERNAL
from treebuild.types import bindingdb, pubchem, pic50

# for the static server
import SimpleHTTPServer
import SocketServer

# directory for chemtree
CHEMTREE_DIR = "/home/ChemTree"

# # test data
input_file = "./drug_class_test.txt"
out_file = "./drug_class_test.json"
properties = {"activities": DEFAULT_ACTIVITY_TYPES, "properties": DEFAULT_PROPERTY_TYPES, "ext_links": []}
treebuild.TreeBuild(input_file, out_file, id_column="index", fps=DEFAULT_FINGERPRINT_TYPES, properties=properties)

#properties = {"activities": DEFAULT_ACTIVITY_TYPES, "properties": [], "ext_links": []}
#treebuild.TreeBuild(input_file, out_file, id_column="index", fps = [] , properties=properties)

# # move the image directory to frontend
for fname in ["./drug_class_test.json"]:
    try:
        shutil.copy(fname, os.path.join(CHEMTREE_DIR, "frontend/dist/data/"))
    except:
        continue
if os.path.exists(os.path.join(CHEMTREE_DIR, "frontend/dist/images")):
    shutil.rmtree(os.path.join(CHEMTREE_DIR, "frontend/dist/images"))
copy_tree("./images", os.path.join(CHEMTREE_DIR, "frontend/dist/images"))

##
# setup the server
##
# change working directory
os.chdir(os.path.join(CHEMTREE_DIR, "frontend"))

PORT = 8000

Handler = SimpleHTTPServer.SimpleHTTPRequestHandler

httpd = SocketServer.TCPServer(("", PORT), Handler)

print "Serving at port", PORT
print "Please open the following link in your browser:"
print "localhost:8000/dist/#/" + out_file.split("/")[-1].split(".")[0]

httpd.serve_forever()
