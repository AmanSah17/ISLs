from api import app
for route in app.routes:
    methods = getattr(route, 'methods', None)
    print(f"{route.path} {methods}")
