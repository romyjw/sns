import hakowan as hkw
n=20
bases=[]
config = hkw.config()
config.z_up()
config.sensor.location = [-2, -3, -1]
# Generate 4K rendering.
config.film.width = 3840
config.film.height = 2160

	
base =  hkw.layer('../data/art/SPINOSAURUS.ply')

lines = hkw.layer('../data/art/crossfield_max_dir.obj').material("Principled", color='#000000')


hkw.render(base+lines, config, filename='../data/art/lineyspiney.png')
	
	
