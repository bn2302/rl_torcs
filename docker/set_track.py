#!/usr/bin/python

import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser(description='Track')
parser.add_argument('-t', '--track')

args = parser.parse_args()

xmlfile = '/usr/local/share/games/torcs/config/raceman/practice.xml'
tree = ET.parse(xmlfile)

root = tree.getroot()
d = root[1][1][0].attrib
d['val'] = args.track
root[1][1][0].attrib = d
tree.write(xmlfile)
