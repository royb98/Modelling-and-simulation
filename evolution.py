import numpy as np
import numpy.random as rn
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
import seaborn as sbn
sbn.set_style('darkgrid', {'legend.frameon': True})         # plotting style


# initial conditions
n_frames = 500                      # number of time steps
food_availability = 500             # maximum number of food (average number of food will be roughly half of this)
bounds = 300                        # size of map
capture_distance = bounds / 150     # distance from which food or prey is captured
mutation_speed = 0.2                # speed with which genome mutates, sets the standard deviation
m_min = 0.3                         # minimum mass of organisms
v_min = 1                           # minimum speed of organisms (speed when not chased)


def distance(x1, x2, uv=False):
    """Distance or direction unit vector between two points, with periodic boundaries"""
    x = x1 - x2

    # if the other point is on the other side of the map, take the opposite direction (across the boundary)
    x = np.where(np.abs(x) > bounds, - np.sign(x) * (2 * bounds - x), x)
    x = x.T

    if uv is True:      # return unit vector (direction)
        return - x / np.sqrt(x[0]**2 + x[1]**2)
    else:               # return distance
        return np.sqrt(x[0]**2 + x[1]**2)


def update(i, fig, scat, data, k, title):
    """Update scatter plot for animation in plot_movie()."""
    scat.set_offsets(data[i][k])
    title.set_text(u'time = {:05}'.format(i))
    return scat


def plot_movie(pos):
    """Show movie of positions of organisms over time."""
    fig, ax = plt.subplots()
    title = ax.text(0.5, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                    transform=ax.transAxes, ha="center")
    ax.set_xlim(-bounds, bounds)
    ax.set_ylim(-bounds, bounds)
    scat_f = ax.scatter(0, 0, c='green', s=2)
    anim_f = animation.FuncAnimation(fig, update, fargs=(fig, scat_f, pos, 0, title), frames=n_frames, interval=20)
    scat_y = ax.scatter(0, 0, c='darkslategrey', s=15)
    anim_y = animation.FuncAnimation(fig, update, fargs=(fig, scat_y, pos, 1, title), frames=n_frames, interval=20)
    scat_p = ax.scatter(0, 0, c='red', s=35)
    anim_p = animation.FuncAnimation(fig, update, fargs=(fig, scat_p, pos, 2, title), frames=n_frames, interval=20)
    ax.set_facecolor('lightgreen')
    ax.grid(b=False)
    plt.show()


def plot_numbers():
    """Make plots showing the number of organisms of each species over time."""
    t_list = range(n_frames)
    fig, ax = plt.subplots(figsize=(8, 1.5), dpi=100)
    ax.plot(t_list, len_f, c='green', label='food')
    ax.plot(t_list, len_y, c='blue', label='prey')
    ax.plot(t_list, len_p, c='red', label='predator')
    ax.legend(loc='upper left', framealpha=0.7)
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1.0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)
    ax.set_xlabel('time step')
    ax.set_ylabel('number of organisms')
    plt.title('Number of organisms over time')
    plt.grid()
    plt.show()


def plot_genome():
    """Make plots describing the evolution of the genome and the distribution at the end of the simulation."""
    fig, ax = plt.subplots(1, 3, figsize=(8, 2), dpi=100)
    t_list = range(n_frames)

    # speed over time
    ax[0].plot(t_list, np.abs(v_predator), c='red', label='predator')
    ax[0].plot(t_list, np.abs(v_prey), c='blue', label='prey')

    # mass over time
    ax[1].plot(t_list, np.abs(m_predator), c='red', label='predator')
    ax[1].plot(t_list, np.abs(m_prey), c='blue', label='prey')

    # speed vs mass of last snapshot
    ax[2].scatter(np.abs(vf_prey), np.abs(mf_prey), c='blue', s=5, label='prey')
    ax[2].scatter(np.abs(vf_predator), np.abs(mf_predator), c='red', s=5, label='predator')

    ax[2].set_xlabel(r'$ v_f $')
    ax[0].set_ylabel(r'$\langle v \rangle$')
    ax[1].set_ylabel(r'$\langle M \rangle$')
    ax[2].set_ylabel(r'$ M_f$')
    for i in range(2):
        ax[i].set_xlabel('time step')
    for a in ax:
        a.legend(loc='upper left', framealpha=0.7)
        a.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        a.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        a.grid(b=True, which='major', color='w', linewidth=1)
        a.grid(b=True, which='minor', color='w', linewidth=.5)
    plt.tight_layout()
    plt.show()


def avg_trait(x):
    """Returns the average genome for a certain species."""
    v = np.mean([x[i].vel for i in range(len(x))])
    m = np.mean([x[i].mass for i in range(len(x))])
    return v, m


class Organism:
    """Class that defines any organism. The genome defines all animals with a speed and weight attribute."""
    def __init__(self, position, genome):
        self.position, self.genome = position, genome
        self.vel = self.genome[0] + np.sign(self.genome[0]) * v_min
        self.mass = self.genome[1] + m_min

        self.wait = 0                                   # timer to stop moving for a while (e.g. after eating)
        self.species, self.Species = None, None         # list of organisms of own species and class of species
        self.target, self.hunters = None, []            # current target or hunters
        self.maxdist = np.sqrt(2) * bounds              # maximum distance at which hunters become interested in me
        self.dead = False

    def seek(self, species):
        """
        Seek nearest target that self can get to first or second (distance is smaller than maxdist).
        Only targets that are lighter than self are considered.
        """
        sel, sel_pos = [], []       # list of possible targets and their positions
        try:
            for i in species:
                if i.mass <= self.mass:                   # if it is lighter than self
                    d = distance(self.position, i.position)
                    if d < i.maxdist:                                   # if self can catch it first or second
                        sel.append(i), sel_pos.append(d)                    # it is a possible target
                elif i.dead is True:                                 # else, eat dead prey
                    d = distance(self.position, i.position)
                    sel.append(i), sel_pos.append(d)
            self.target = sel[np.argmin(np.asarray(sel_pos))]       # take nearest of possible targets
        except:
            self.target = False

    def move(self, target, chase=False):
        """Change position of self to move towards food or flee from hunter."""
        speed = v_min if chase is False \
            else self.vel       # default speed is 1, chasing speed is given by genome

        # if positioned outside the boundary, reposition to the other end of the map
        for i in [0, 1]:
            if abs(self.position[i]) > bounds:
                self.position[i] -= np.sign(self.position[i]) * 2 * bounds

        # move and use up energy
        if self.wait > 0 and chase is False:    # except when waiting (e.g. after eating or when starving)
            self.wait -= 1
        else:
            self.wait = 0
            direction = distance(self.position, target.position, uv=True)
            self.position += speed * direction

            # kinetic energy lost
            self.energy -= self.mass**(2/3) * (2 * np.abs(speed) + 0.1 * speed**4)

    def eat(self, target, nutrition):
        """Eat prey if within capture distance. If another hunter is nearby, share the energy."""
        if distance(target.position, self.position) <= capture_distance:
            if target.maxdist < 2:      # if two hunters of the target are within a distance of 2, share the target
                for h in target.hunters:
                    h.energy += nutrition / 2
                    h.wait += 10
            else:                       # otherwise, eat target
                self.energy += nutrition
                self.wait = 10

            # remove target
            target.species.remove(target)
            del target

    def reproduce(self):
        """Produce a copy of self with a slightly different genome."""
        if self.energy >= 2.5 * self.energy0:           # if self has enough energy, reproduce
            # produce new related genome
            new_speed = self.genome[0] * np.abs(rn.normal(1, mutation_speed))
            new_mass = self.genome[1] * np.abs(rn.normal(1, mutation_speed))
            new_genome = [new_speed, new_mass]

            # produce child and add to relatives
            child = self.Species(copy.deepcopy(self.position), new_genome)
            self.species.append(child)
            self.energy -= 1.5 * self.energy0           # reproduction cost

    def die(self):
        """Completely removes self."""
        self.species.remove(self)
        del self

    def md(self, species):
        """
        Defines maximum distance from which hunting species is interested in self.
        If there are two hunters, a third hunter has to be closer than the second hunter in order to be interested.
        This ensures that no more than two hunters go after the same target, and they should be closeby.
        """
        self.hunters, p = [], []
        for i in species:           # make list of hunters
            if i.target == self:
                p.append(i.position)
                self.hunters.append(i)
        if len(self.hunters) <= 1:             # if only one wants the target, don't set a maximum distance
            self.maxdist = np.sqrt(2) * bounds
        else:                       # else, set a maximum distance for a new 3rd hunter
            self.maxdist = np.sort(distance(self.position, np.array(p)))[1] + 0.0001


class Food(Organism):
    """Stationary organism which can be eaten by prey."""
    def __init__(self, position):
        super().__init__(position, [0, 0])
        self.position = position
        self.species = food                 # class of self
        self.Species = Food                 # chase cooldown, continue to move just after a chase
        self.maxdist = np.sqrt(2) * bounds  # maximum distance at which hunters become interested in self

    def action(self):
        self.md(prey)       # calculate self.maxdist


class Prey(Organism):
    """Moving organism which eats food and can be eaten by predator."""
    def __init__(self, *args):
        super().__init__(*args)
        self.energy0 = 150                  # initial energy
        self.energy = self.energy0
        self.species = prey                 # list of objects that self belongs to
        self.Species = Prey                 # class of self
        self.chase_cd = 0                   # chase cooldown, continue to move just after a chase
        self.dead_cd = 500                   # number of steps before a dead self is removed
        self.maxdist = np.sqrt(2) * bounds  # maximum distance at which hunters become interested in self
        self.dead = False                   # if dead == True, self can be eaten by any predator no matter what

    def action(self):
        """Defines which actions to take each time step."""
        self.md(predator)       # calculate self.maxdist

        self.energy -= .2       # deduce some energy for living
        if self.energy < 0:     # if dead, don't do anything (self still exists so it can be eaten by predators)
            if self.dead is True:
                if self.dead_cd == 0:
                    self.die()
                else:
                    self.dead_cd -= 1
                    return
            else:
                self.dead = True

        # compute distances to predators
        predator_distances = [distance(self.position, p.position) for p in predator]
        chaser = predator[np.argmin(predator_distances)]
        if min(predator_distances) <= 4:    # flee from predator if it is near
            self.move(chaser, chase=True)
            self.chase_cd = 10
            return
        elif self.chase_cd > 0:             # if just finished a chase, continue fleeing from chaser for a while
            self.move(chaser, chase=True)
            self.chase_cd -= 1
            return

        # graze closest food
        self.seek(food)
        if self.target is False:            # if there is no food, kill self
            self.energy = -1000
            return
        self.move(self.target)              # otherwise, move to target
        self.eat(self.target, 160)

        self.reproduce()                    # if there is enough energy, make a child


class Predator(Organism):
    """Moving organism which eats prey."""
    def __init__(self, *args):
        super().__init__(*args)
        self.energy0 = 350
        self.energy = self.energy0
        self.species = predator     # list of objects that self belongs tp
        self.Species = Predator     # class of self

    # decide what to do:
    def action(self):
        self.energy -= .2           # deduce some energy for living
        if self.energy <= 0:        # if out of energy, remove self
            self.die()
            return

        # hunt closest prey
        self.seek(prey)
        if self.target is False:        # if there are no prey, kill self
            self.die()
            return
        if distance(self.position, self.target.position) <= 4:      # start to chase target if close
            self.move(self.target, chase=True)
        else:                                                       # otherwise, move to target with normal speed
            self.move(self.target)

        self.eat(self.target, 320)      # if close enough, eat the target

        self.reproduce()                # if there is enough energy, make a child


# initiate food, prey and predators
food, prey, predator = [], [], []       # lists of animals
for _ in range(80):
    rn_pos = rn.uniform(-bounds, bounds, 2)                             # random position
    food.append(Food(rn_pos))                                           # make food
for _ in range(70):
    rn_pos = rn.uniform(-bounds, bounds, 2)                             # random position
    rn_gen = np.array([-rn.normal(1.2, 0.2), rn.normal(.5, 0.2)])       # random genome (velocity, mass)
    prey.append(Prey(rn_pos, rn_gen))                                   # make prey
for _ in range(25):
    rn_pos = rn.uniform(-bounds, bounds, 2)                             # random position
    rn_gen = np.array([rn.normal(1.5, 0.2), rn.normal(.7, 0.2)])        # random genome (velocity, mass)
    predator.append(Predator(rn_pos, rn_gen))                           # make predator


# integrate
pos_list = []       # list of positions, for each frame, species and individual
len_f, len_y, len_p = np.zeros((3, n_frames))                       # list of number of organisms over time
v_prey, m_prey, v_predator, m_predator, = np.zeros((4, n_frames))   # velocity and mass of prey and predator
for time in range(n_frames):
    try:
        # spawn food at random location, with a chance of food_spawn depending on the current amount of food
        food_spawn = (food_availability - len(food)) / food_availability
        if rn.random() < food_spawn:
            food.append(Food(rn.uniform(-bounds, bounds, 2)))

        # let every organism carry out their actions
        for _ in food:
            _.action()
        for _ in prey:
            _.action()
        for _ in predator:
            _.action()

        # print progress and number of organisms
        if time % 200 == 0:
            print('time = {}, n_prey = {}, n_predator = {}'.format(time, len(prey), len(predator)))
    except Exception as e:
        n_frames = time - 1
        print('ERROR: ' + str(e))
        break

    # save traits of prey and predators
    v_prey[time], m_prey[time] = avg_trait(prey)
    v_predator[time], m_predator[time] = avg_trait(predator)
    len_f[time], len_y[time], len_p[time] = len(food), len(prey), len(predator)
    new_pos = [[individual.position for individual in species] for species in [food, prey, predator]]
    pos_list.append(copy.deepcopy(new_pos))
print('DONE')


# append genome values of all prey and predators at final time
vf_prey = [i.vel for i in prey]
mf_prey = [i.mass for i in prey]

vf_predator = [i.vel for i in predator]
mf_predator = [i.mass for i in predator]


# make plots and movies
plot_genome()
plot_numbers()
plot_movie(pos_list)
