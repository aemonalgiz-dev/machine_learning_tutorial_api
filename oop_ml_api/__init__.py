"""A thin HTTP layer over the oop_ml library.

The teaching website computes against this: small numeric inputs arrive as JSON,
become oop_ml ``Feature`` objects and a model, and the fitted answer goes back as
JSON. Nothing here reimplements any mathematics -- the library owns all of it,
and this layer is a translator plus a bouncer at the door.
"""
