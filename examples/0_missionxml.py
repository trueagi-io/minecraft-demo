import tagilmo.utils.mission_builder as mb

# we can specify separate parameters through nested constructors
miss = mb.MissionXML(serverSection=mb.ServerSection(initial_conditions=mb.ServerInitialConditions(weather_string="rain")))

# we can use a nicer API, though
miss.setWorld(mb.flatworld(""))
miss.addAgent()
miss.setObservations(mb.Observations(bAll=False, bRay=True, bFullStats=True))

# we can also set some parameters this way if they are lacking API
miss.agentSections[0].agentstart.inventory = ["diamond_pickaxe", "wooden_axe"]

print(miss.xml())
print("================\n")
print(miss.getAgentNames())
