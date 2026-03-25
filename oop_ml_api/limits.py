"""The bounds on what a request may ask this API to compute.

The website is a teaching tool with small, hand-entered examples, so the inputs
are small by design. Enforcing that at the door is not only about cost -- a
bounded fit is a fast fit, and a teaching widget that stays responsive teaches
better -- it is also the API's half of the library's own principle that inputs
are well defined rather than arbitrary. A request outside these bounds is a
typed 422 with the limit named, never a slow compute or an out-of-memory.
"""

MAX_POINTS = 100
"""The most data points a single request may carry.

Two is the minimum a line can be fitted through; one hundred is already far more
than a person drags into a teaching widget by hand, and well below anything that
would make the fit slow. The ceiling exists so a crafted request cannot turn a
teaching endpoint into a compute job, and the widget enforces the same number on
its own side so a learner meets the limit before the request is ever sent.
"""

MIN_POINTS = 2
"""The fewest points a fit needs to mean anything.

A line through one point is undetermined, and the library says so; the API
refuses it a layer earlier, with a message a learner can read, rather than
letting the underlying ``TooFewValuesError`` surface as the explanation.
"""

MAX_MAGNITUDE = 1_000_000.0
"""How large any single coordinate may be, in absolute value.

Not a precision limit -- the library handles large magnitudes correctly -- but a
sanity bound on hand-entered teaching data, and a guard against a value crafted
to overflow something downstream. A teaching example lives comfortably inside a
few orders of magnitude.
"""
