#!/usr/bin/env python
from tft import app
import os

app.run(
    host=os.getenv('LISTEN','0.0.0.0'),
    port=int(os.getenv('PORT','80')),
    debug=False,
    threaded=True
)
