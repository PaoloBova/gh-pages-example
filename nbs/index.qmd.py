"""---
title: Home
pagetitle: Phd Showcase
page-layout: custom
section-divs: false
css: index.css
toc: false
image: https:/paolobova.github.io/gh-pages-example/images/placeholder.png
description: Demos, analyses, writing, and documentation for Paolo Bova's Phd work.
---"""

from fastcore.foundation import L
from nbdev import qmd

def img(fname, classes=None, **kwargs): return qmd.img(f"images/{fname}", classes=classes, **kwargs)
def btn(txt, link): return qmd.btn(txt, link=link, classes=['btn-action-primary', 'btn-action', 'btn', 'btn-success', 'btn-lg'])
def banner(txt, classes=None, style=None): return qmd.div(txt, L('hero-banner')+classes, style=style)

def feature(im, desc): return qmd.div(f"{img(im+'.svg')}\n\n{desc}\n", ['feature', 'g-col-12', 'g-col-sm-6', 'g-col-md-4'])

def b(*args, **kwargs): print(banner (*args, **kwargs))
def d(*args, **kwargs): print(qmd.div(*args, **kwargs))

b(f"""## Welcome to my Phd showcase

{btn('See documentation', '/methods.html')}""", 'content-block', style={"margin-top": "40px"})