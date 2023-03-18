from engine.main import Engine
import json

engine = Engine()

f_app = open('SecureWebContainer.json')
application = json.load(f_app)

f_offers = open('offers_20.json')
offers = json.load(f_offers)

engine.solve(application, offers)
