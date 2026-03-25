"""One module per teaching concept, each a self-contained FastAPI router.

The library groups its code by the task; this mirrors it, one router per
concept the website teaches. A concept's module owns its request and response
shapes and its single endpoint, so adding a concept to the site is adding a
module here and mounting its router -- the same "a new thing is a new file, not
an edit to a central switch" the library holds to.
"""
