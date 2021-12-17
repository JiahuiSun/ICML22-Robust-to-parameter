import contextlib
import os
import tempfile

import xml.etree.ElementTree as ET

import roboschool
from roboschool.gym_mujoco_walkers import (
    RoboschoolForwardWalkerMujocoXML, RoboschoolHalfCheetah, RoboschoolHopper, RoboschoolWalker2d, RoboschoolHumanoid, RoboschoolAnt
)
from roboschool.gym_pendulums import (
    RoboschoolInvertedPendulum, RoboschoolInvertedDoublePendulum)
from roboschool.gym_reacher import RoboschoolReacher

from .base import EnvBinarySuccessMixin
from .classic_control import uniform_exclude_inner

# Determine Roboschool asset location based on its module path.
ROBOSCHOOL_ASSETS = os.path.join(roboschool.__path__[0], 'mujoco_assets')


class RoboschoolTrackDistSuccessMixin(EnvBinarySuccessMixin):
    """Treat reaching certain distance on track as a success."""

    def is_success(self):
        """Returns True is current state indicates success, False otherwise
        TODO: 我猜作者的意思是，horizon最多1000，agent永远走不到终点，也就无法success
         x=100 correlates to the end of the track on Roboschool,
         but with the default 1000 max episode length most (all?) agents
         won't reach it (DD PPO2 Hopper reaches ~40), so we use something lower
         """
        target_dist = 20
        if self.robot_body.pose().xyz()[0] >= target_dist:
             #print("[SUCCESS]: xyz is {}, reached x-target {}".format(
             #      self.robot_body.pose().xyz(), target_dist))
            return True
        else:
             #print("[NO SUCCESS]: xyz is {}, x-target is {}".format(
             #      self.robot_body.pose().xyz(), target_dist))
            return False


class RoboschoolXMLModifierMixin:
    """Mixin with XML modification methods."""
    @contextlib.contextmanager
    def modify_xml(self, asset):
        """Context manager allowing XML asset modifcation."""

        # tree = ET.ElementTree(ET.Element(os.path.join(ROBOSCHOOL_ASSETS, asset)))
        tree = ET.parse(os.path.join(ROBOSCHOOL_ASSETS, asset))
        yield tree

        # Create a new temporary .xml file
        # mkstemp returns (int(file_descriptor), str(full_path))
        fd, path = tempfile.mkstemp(suffix='.xml')
        # Close the file to prevent a file descriptor leak
        # See: https://www.logilab.org/blogentry/17873
        # We can also wrap tree.write in 'with os.fdopen(fd, 'w')' instead
        os.close(fd)
        tree.write(path)

        # Delete previous file before overwriting self.model_xml
        if os.path.isfile(self.model_xml):
            os.remove(self.model_xml)
        self.model_xml = path

        # Original fix using mktemp:
        # mktemp (depreciated) returns str(full_path)
        #   modified_asset = tempfile.mktemp(suffix='.xml')
        #   tree.write(modified_asset)
        #   self.model_xml = modified_asset

    def __del__(self):
        """Deletes last remaining xml files after use"""
        # (Note: this won't ensure the final tmp file is deleted on a crash/SIGBREAK/etc.)
        if os.path.isfile(self.model_xml):
            os.remove(self.model_xml)


# =============== Reacher ===================
class ModifiableRoboschoolReacher(RoboschoolReacher, RoboschoolTrackDistSuccessMixin):
    RANDOM_LOWER_BODY_SIZE = 0.008
    RANDOM_UPPER_BODY_SIZE = 0.05

    RANDOM_LOWER_BODY_LENGTH = 0.1
    RANDOM_UPPER_BODY_LENGTH = 0.13

    def _reset(self, new=True):
        return super(ModifiableRoboschoolReacher, self)._reset()

    @property
    def parameters(self):
        return {'id': self.spec.id, }


class RandomNormalRoboschoolReacher(RoboschoolXMLModifierMixin, ModifiableRoboschoolReacher):
    def randomize_env(self):
        self.size = self.np_random.uniform(
            self.RANDOM_LOWER_BODY_SIZE, self.RANDOM_UPPER_BODY_SIZE)
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_BODY_LENGTH, self.RANDOM_UPPER_BODY_LENGTH)

        with self.modify_xml('reacher.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('fromto', "0 0 0 " + str(self.length) + " 0 0")
                if elem.attrib['name'] == "link0":
                    elem.set('size', str(self.size))
            for elem in tree.iterfind('worldbody/body/body'):
                elem.set('pos', str(self.length) + " 0 0")
            for elem in tree.iterfind('worldbody/body/body/geom'):
                elem.set('size', str(self.size))

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalRoboschoolReacher, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalRoboschoolReacher, self).parameters
        parameters.update({'size': self.size, 'length': self.length, })
        return parameters


# ============== InvertedPendulum ===============
class ModifiableRoboschoolInvertedPendulum(RoboschoolInvertedPendulum, RoboschoolTrackDistSuccessMixin):

    RANDOM_LOWER_CART_SIZE = 0.05
    RANDOM_UPPER_CART_SIZE = 0.25

    RANDOM_LOWER_POLE_LENGTH = 0.5
    RANDOM_UPPER_POLE_LENGTH = 2.0

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1

    RANDOM_LOWER_POLE_SIZE = 0.03
    RANDOM_UPPER_POLE_SIZE = 0.068

    RANDOM_LOWER_RAIL_SIZE = 0.01
    RANDOM_UPPER_RAIL_SIZE = 0.03

    def _reset(self, new=True):
        return super(ModifiableRoboschoolInvertedPendulum, self)._reset()

    @property
    def parameters(self):
        return {'id': self.spec.id, }


class RandomNormalInvertedPendulum(RoboschoolXMLModifierMixin, ModifiableRoboschoolInvertedPendulum):
    def randomize_env(self):
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_POLE_LENGTH, self.RANDOM_UPPER_POLE_LENGTH)
        self.cartsize = self.np_random.uniform(
            self.RANDOM_LOWER_CART_SIZE, self.RANDOM_UPPER_CART_SIZE)

        with self.modify_xml('inverted_pendulum.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('size', str(self.cartsize) + ' ' + str(self.cartsize))
            for elem in tree.iterfind('worldbody/body/body/geom'):
                elem.set('fromto', "0 0 0 0.001 0 " + str(self.length))

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalInvertedPendulum, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalInvertedPendulum, self).parameters
        parameters.update({'length': self.length, 'cart-size': self.cartsize})
        return parameters


# random pole size
class RandomPolesizeInvertedPendulum(RoboschoolXMLModifierMixin, ModifiableRoboschoolInvertedPendulum):
    def randomize_env(self):  # random pole size
        # self.cartsize = self.np_random.uniform(self.RANDOM_LOWER_CART_SIZE, self.RANDOM_UPPER_CART_SIZE)
        # self.length = self.np_random.uniform(self.RANDOM_LOWER_POLE_LENGTH, self.RANDOM_UPPER_POLE_LENGTH)

        # self.friction = self.np_random.uniform(self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        # self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)
        self.polesize = self.np_random.uniform(
            self.RANDOM_LOWER_POLE_SIZE, self.RANDOM_UPPER_POLE_SIZE)

        with self.modify_xml('inverted_pendulum.xml') as tree:
            # for elem in tree.iterfind('worldbody/body/geom'):
            #     elem.set('size', str(self.cartsize) + ' ' + str(self.cartsize))
            for elem in tree.iterfind('worldbody/body/body/geom'):
                # elem.set('fromto', "0 0 0 0.001 0 " + str(self.length))
                elem.set('size', str(self.polesize)+" 0.068")

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomPolesizeInvertedPendulum, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomPolesizeInvertedPendulum, self).parameters
        parameters.update({'pole-size': self.polesize, })
        return parameters


class RandomPolesizeInvertedPendulumEval(RoboschoolXMLModifierMixin, ModifiableRoboschoolInvertedPendulum):
    # def randomize_env(self): # random pole size
    #     self.polesize = self.np_random.uniform(self.RANDOM_LOWER_POLE_SIZE, self.RANDOM_UPPER_POLE_SIZE)
    #     with self.modify_xml('inverted_pendulum.xml') as tree:
    #         for elem in tree.iterfind('worldbody/body/body/geom'):
    #             elem.set('size', str(self.polesize)+" 0.068")
    def set_envparam(self, *args):
        self.polesize = args[0]
        with self.modify_xml('inverted_pendulum.xml') as tree:
            for elem in tree.iterfind('worldbody/body/body/geom'):
                elem.set('size', str(self.polesize)+" 0.068")

    def _reset(self, new=True):
        if new:
            with self.modify_xml('inverted_pendulum.xml') as tree:
                for elem in tree.iterfind('worldbody/body/body/geom'):
                    elem.set('size', str(self.polesize) + " 0.068")
        return super(RandomPolesizeInvertedPendulumEval, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomPolesizeInvertedPendulumEval, self).parameters
        parameters.update({'pole-size': self.polesize, })
        return parameters


# random pole len
class RandomPolelenInvertedPendulum(RoboschoolXMLModifierMixin, ModifiableRoboschoolInvertedPendulum):
    def randomize_env(self):  # random pole size
        # self.cartsize = self.np_random.uniform(self.RANDOM_LOWER_CART_SIZE, self.RANDOM_UPPER_CART_SIZE)
        # self.length = self.np_random.uniform(self.RANDOM_LOWER_POLE_LENGTH, self.RANDOM_UPPER_POLE_LENGTH)

        # self.friction = self.np_random.uniform(self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        # self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_POLE_LENGTH, self.RANDOM_UPPER_POLE_LENGTH)

        with self.modify_xml('inverted_pendulum.xml') as tree:
            # for elem in tree.iterfind('worldbody/body/geom'):
            #     elem.set('size', str(self.cartsize) + ' ' + str(self.cartsize))
            for elem in tree.iterfind('worldbody/body/body/geom'):
                elem.set('fromto', "0 0 0 0.001 0 " + str(self.length))
                # elem.set('size', str(self.polesize)+" 0.068")

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomPolelenInvertedPendulum, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomPolelenInvertedPendulum, self).parameters
        parameters.update({'length': self.length, })
        return parameters


class RandomPolelenInvertedPendulumEval(RoboschoolXMLModifierMixin, ModifiableRoboschoolInvertedPendulum):
    # def randomize_env(self): # random pole size
    #     self.polesize = self.np_random.uniform(self.RANDOM_LOWER_POLE_SIZE, self.RANDOM_UPPER_POLE_SIZE)
    #     with self.modify_xml('inverted_pendulum.xml') as tree:
    #         for elem in tree.iterfind('worldbody/body/body/geom'):
    #             elem.set('size', str(self.polesize)+" 0.068")
    def set_envparam(self, *args):
        self.length = args[0]
        with self.modify_xml('inverted_pendulum.xml') as tree:
            for elem in tree.iterfind('worldbody/body/body/geom'):
                elem.set('fromto', "0 0 0 0.001 0 " + str(self.length))

    def _reset(self, new=True):
        if new:
            with self.modify_xml('inverted_pendulum.xml') as tree:
                for elem in tree.iterfind('worldbody/body/body/geom'):
                    elem.set('fromto', "0 0 0 0.001 0 " + str(self.length))
        return super(RandomPolelenInvertedPendulumEval, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomPolelenInvertedPendulumEval, self).parameters
        parameters.update({'length': self.length, })
        return parameters


# consider 3d environment parameters
class RandomNormalInvertedPendulum3DEnvParam(RoboschoolXMLModifierMixin, ModifiableRoboschoolInvertedPendulum):
    def randomize_env(self):
        self.cartsize = self.np_random.uniform(
            self.RANDOM_LOWER_CART_SIZE, self.RANDOM_UPPER_CART_SIZE)
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_POLE_LENGTH, self.RANDOM_UPPER_POLE_LENGTH)
        # self.friction = self.np_random.uniform(self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        # self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)
        self.polesize = self.np_random.uniform(
            self.RANDOM_LOWER_POLE_SIZE, self.RANDOM_UPPER_POLE_SIZE)

        with self.modify_xml('inverted_pendulum.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('size', str(self.cartsize) + ' ' + str(self.cartsize))
            for elem in tree.iterfind('worldbody/body/body/geom'):
                elem.set('fromto', "0 0 0 0.001 0 " + str(self.length))
                elem.set('size', str(self.polesize)+" 0.068")

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalInvertedPendulum3DEnvParam, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(
            RandomNormalInvertedPendulum3DEnvParam, self).parameters
        parameters.update(
            {'length': self.length, 'pole-size': self.polesize, 'cart-size': self.cartsize})
        return parameters


class RandomNormalInvertedPendulum3DParamEvaluate(RoboschoolXMLModifierMixin, ModifiableRoboschoolInvertedPendulum):

    def set_envparam(self, *args):
        print(args)
        self.length = args[0]
        self.cartsize = args[1]
        self.polesize = args[2]

        with self.modify_xml('inverted_pendulum.xml') as tree:

            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('size', str(self.cartsize) + ' ' + str(self.cartsize))
            for elem in tree.iterfind('worldbody/body/body/geom'):
                elem.set('fromto', "0 0 0 0.001 0 " + str(self.length))
                elem.set('size', str(self.polesize)+" 0.068")

    def _reset(self, new=True):
        if new:
            # self.randomize_env()
            print('==============in invertedpendulum reset ==========')
            print(self.length)
            print(self.cartsize)
            print(self.polesize)
            with self.modify_xml('inverted_pendulum.xml') as tree:
                for elem in tree.iterfind('worldbody/body/geom'):
                    elem.set('size', str(self.cartsize) +
                             ' ' + str(self.cartsize))
                for elem in tree.iterfind('worldbody/body/body/geom'):
                    elem.set('fromto', "0 0 0 0.001 0 " + str(self.length))
                    elem.set('size', str(self.polesize)+" 0.068")

        return super(RandomNormalInvertedPendulum3DParamEvaluate, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(
            RandomNormalInvertedPendulum3DParamEvaluate, self).parameters
        parameters.update({'length': self.length, 'cart-size': self.cartsize})


# consider 4d environment parameters
class RandomNormalInvertedPendulum4DEnvParam(RoboschoolXMLModifierMixin, ModifiableRoboschoolInvertedPendulum):
    def randomize_env(self):
        self.cartsize = self.np_random.uniform(
            self.RANDOM_LOWER_CART_SIZE, self.RANDOM_UPPER_CART_SIZE)
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_POLE_LENGTH, self.RANDOM_UPPER_POLE_LENGTH)
        # self.friction = self.np_random.uniform(self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        # self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)
        self.polesize = self.np_random.uniform(
            self.RANDOM_LOWER_POLE_SIZE, self.RANDOM_UPPER_POLE_SIZE)
        self.railsize = self.np_random.uniform(
            self.RANDOM_LOWER_RAIL_SIZE, self.RANDOM_UPPER_RAIL_SIZE)

        with self.modify_xml('inverted_pendulum.xml') as tree:
            for elem in tree.iterfind('worldbody/geom'):
                # print(elem.get('name'))
                elem.set('size', str(self.railsize)+" 2")
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('size', str(self.cartsize) + ' ' + str(self.cartsize))
            for elem in tree.iterfind('worldbody/body/body/geom'):
                elem.set('fromto', "0 0 0 '0.001 0 " + str(self.length))
                elem.set('size', str(self.polesize)+" 0.068")

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalInvertedPendulum4DEnvParam, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(
            RandomNormalInvertedPendulum4DEnvParam, self).parameters
        parameters.update(
            {'length': self.length, 'pole-size': self.polesize, 'cart-size': self.cartsize})
        return parameters


class RandomNormalInvertedPendulum4DParamEvaluate(RoboschoolXMLModifierMixin, ModifiableRoboschoolInvertedPendulum):

    def set_envparam(self, *args):
        print(args)
        self.length = args[0]
        self.cartsize = args[1]
        self.polesize = args[2]
        self.railsize = args[3]

        with self.modify_xml('inverted_pendulum.xml') as tree:
            for elem in tree.iterfind('worldbody/geom'):
                # print(elem.get('name'))
                elem.set('size', str(self.railsize)+" 2")
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('size', str(self.cartsize) + ' ' + str(self.cartsize))
            for elem in tree.iterfind('worldbody/body/body/geom'):
                elem.set('fromto', "0 0 0 0.001 0 " + str(self.length))
                elem.set('size', str(self.polesize)+" 0.068")

    def _reset(self, new=True):
        if new:
            # self.randomize_env()
            print('==============in invertedpendulum reset ==========')
            print(self.length)
            print(self.cartsize)
            print(self.polesize)
            with self.modify_xml('inverted_pendulum.xml') as tree:
                for elem in tree.iterfind('worldbody/geom'):
                    # print(elem.get('name'))
                    elem.set('size', str(self.railsize)+" 2")
                for elem in tree.iterfind('worldbody/body/geom'):
                    elem.set('size', str(self.cartsize) +
                             ' ' + str(self.cartsize))
                for elem in tree.iterfind('worldbody/body/body/geom'):
                    elem.set('fromto', "0 0 0 0.001 0 " + str(self.length))
                    elem.set('size', str(self.polesize)+" 0.068")

        return super(RandomNormalInvertedPendulum4DParamEvaluate, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(
            RandomNormalInvertedPendulum4DParamEvaluate, self).parameters
        parameters.update({'length': self.length, 'cart-size': self.cartsize})


class RandomNormalInvertedPendulumEvaluate(RoboschoolXMLModifierMixin, ModifiableRoboschoolInvertedPendulum):

    def set_envparam(self, cartsize, length):

        self.length = length
        self.cartsize = cartsize

        with self.modify_xml('inverted_pendulum.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('size', str(self.cartsize) + ' ' + str(self.cartsize))
            for elem in tree.iterfind('worldbody/body/body/geom'):
                elem.set('fromto', "0 0 0 0.001 0 " + str(self.length))

    def _reset(self, new=True):
        if new:
            # self.randomize_env()
            print('==============in invertedpendulum reset ==========')
            print(self.length)
            print(self.cartsize)
            with self.modify_xml('inverted_pendulum.xml') as tree:
                for elem in tree.iterfind('worldbody/body/geom'):
                    elem.set('size', str(self.cartsize) +
                             ' ' + str(self.cartsize))
                for elem in tree.iterfind('worldbody/body/body/geom'):
                    elem.set('fromto', "0 0 0 0.001 0 " + str(self.length))

        return super(RandomNormalInvertedPendulumEvaluate, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(
            RandomNormalInvertedPendulumEvaluate, self).parameters
        parameters.update({'length': self.length, 'cart-size': self.cartsize})


# =========== Ant ================
class ModifiableRoboschoolAnt(RoboschoolAnt, RoboschoolTrackDistSuccessMixin):
    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250

    EXTREME_LOWER_DENSITY = 500
    EXTREME_UPPER_DENSITY = 1500

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1

    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 2.5

    RANDOM_LOWER_DAMPING = 0.5
    RANDOM_UPPER_DAMPING = 2.5

    RANDOM_LOWER_FOOTLEN = 0.1
    RANDOM_UPPER_FOOTLEN = 1.8

    def _reset(self, new=True):
        return super(ModifiableRoboschoolAnt, self)._reset()

    @property
    def parameters(self):
        return {'id': self.spec.id, }


class RandomNormalAnt(RoboschoolXMLModifierMixin, ModifiableRoboschoolAnt):
    def randomize_env(self):
        self.density = self.np_random.uniform(
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)

        with self.modify_xml('ant.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalAnt, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalAnt, self).parameters
        parameters.update(
            {'density': self.density, 'friction': self.friction, })
        return parameters


class RandomNormalFDAnt(RoboschoolXMLModifierMixin, ModifiableRoboschoolAnt):
    def randomize_env(self):
        # self.density = self.np_random.uniform(self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        self.damping = self.np_random.uniform(
            self.RANDOM_LOWER_DAMPING, self.RANDOM_UPPER_DAMPING)

        # self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

        with self.modify_xml('ant.xml') as tree:
            for elem in tree.iterfind('default/joint'):
                elem.set('damping', str(self.damping))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalFDAnt, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalFDAnt, self).parameters
        parameters.update(
            {'damping': self.damping, 'friction': self.friction, })
        return parameters


class RandomNormalFootAnt(RoboschoolXMLModifierMixin, ModifiableRoboschoolAnt):
    def randomize_env(self):
        # self.density = self.np_random.uniform(self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        # self.friction = self.np_random.uniform(self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        # self.damping = self.np_random.uniform(self.RANDOM_LOWER_DAMPING, self.RANDOM_UPPER_DAMPING)
        self.front_left_foot = self.np_random.uniform(
            self.RANDOM_LOWER_FOOTLEN, self.RANDOM_UPPER_FOOTLEN)
        self.front_right_foot = - \
            self.np_random.uniform(
                self.RANDOM_LOWER_FOOTLEN, self.RANDOM_UPPER_FOOTLEN)

        # self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

        with self.modify_xml('ant.xml') as tree:
            # for elem in tree.iterfind('default/joint'):
            #     elem.set('damping', str(self.damping))
            # for elem in tree.iterfind('default/geom'):
            #     elem.set('friction', str(self.friction) + ' .1 .1')
            for elem in tree.iterfind('worldbody/body/body/body/body/geom'):
                # print(elem.attrib['name'])
                if elem.attrib['name'] == "left_ankle_geom":
                    # print(elem.attrib['name'])
                    elem.set('fromto', "0.0 0.0 0.0 " +
                             str(self.front_left_foot)+" 0.4 0.0")
                elif elem.attrib['name'] == "right_ankle_geom":
                    # print(elem.attrib['name'])
                    elem.set('fromto', "0.0 0.0 0.0 " +
                             str(self.front_right_foot) + " 0.4 0.0")

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalFootAnt, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalFootAnt, self).parameters
        parameters.update({'front_left_foot': self.front_left_foot,
                           'front_right_foot': self.front_right_foot, })
        return parameters


# =============== Humanoid =============
class ModifiableRoboschoolHumanoid(RoboschoolHumanoid, RoboschoolTrackDistSuccessMixin):
    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 500
    EXTREME_UPPER_DENSITY = 1500

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1
    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 1.4

    def _reset(self, new=True):
        return super(ModifiableRoboschoolHumanoid, self)._reset()

    @property
    def parameters(self):
        return {'id': self.spec.id, }


class RandomNormalHumanoid(RoboschoolXMLModifierMixin, ModifiableRoboschoolHumanoid):
    def randomize_env(self):
        self.density = self.np_random.uniform(
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)

        with self.modify_xml('humanoid_symmetric.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalHumanoid, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalHumanoid, self).parameters
        parameters.update(
            {'density': self.density, 'friction': self.friction, })
        return parameters


class RandomFrictionHumanoid(RoboschoolXMLModifierMixin, ModifiableRoboschoolHumanoid):
    def randomize_env(self):
        # self.density = self.np_random.uniform(self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        # self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

        with self.modify_xml('humanoid_symmetric.xml') as tree:
            # for elem in tree.iterfind('worldbody/body/geom'):
            #     elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomFrictionHumanoid, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomFrictionHumanoid, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomFrictionHumanoidEval(RoboschoolXMLModifierMixin, ModifiableRoboschoolHumanoid):

    def set_envparam(self, friction):
        self.friction = friction
        with self.modify_xml('humanoid_symmetric.xml') as tree:
            # for elem in tree.iterfind('worldbody/body/geom'):
            #     elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            with self.modify_xml('humanoid_symmetric.xml') as tree:
                # for elem in tree.iterfind('worldbody/body/geom'):
                #     elem.set('density', str(self.density))
                for elem in tree.iterfind('default/geom'):
                    elem.set('friction', str(self.friction) + ' .1 .1')
        return super(RandomFrictionHumanoidEval, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomFrictionHumanoidEval, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


# ================ Walker2d =====================
class ModifiableRoboschoolWalker2d(RoboschoolWalker2d, RoboschoolTrackDistSuccessMixin):
    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 500
    EXTREME_UPPER_DENSITY = 1500

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1
    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 1.4

    # RANDOM_LOWER_POWER = 0.7
    # RANDOM_UPPER_POWER = 1.1
    # EXTREME_LOWER_POWER = 0.5
    # EXTREME_UPPER_POWER = 1.3

    def _reset(self, new=True):
        return super(ModifiableRoboschoolWalker2d, self)._reset()

    @property
    def parameters(self):
        return {'id': self.spec.id, }


class RandomNormalWalker2d(RoboschoolXMLModifierMixin, ModifiableRoboschoolWalker2d):
    def set_density(self, density):
        self.density = density
        with self.modify_xml('walker2d.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('density', str(self.density) + ' .1 .1')

    def set_friction(self, friction):
        self.friction = friction
        with self.modify_xml('walker2d.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def randomize_env(self):
        # TODO: 就是这里！！！修改参数
        self.density = self.np_random.uniform(
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)

        with self.modify_xml('walker2d.xml') as tree:
            # for elem in tree.iterfind('worldbody/body/geom'):
            #     elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('density', str(self.density) + ' .1 .1')
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalWalker2d, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalWalker2d, self).parameters
        parameters.update(
            {'density': self.density, 'friction': self.friction, })
        return parameters


class RandomNormalWalker2dEvaluate(RoboschoolXMLModifierMixin, ModifiableRoboschoolWalker2d):

    def set_envparam(self, density, friction):
        self.density = density
        self.friction = friction

        with self.modify_xml('walker2d.xml') as tree:
            # for elem in tree.iterfind('worldbody/body/geom'):
            #     elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('density', str(self.density) + ' .1 .1')
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            # self.randomize_env()
            print('==============in reset ==========')
            print(self.density)
            print(self.friction)
            with self.modify_xml('walker2d.xml') as tree:
                # for elem in tree.iterfind('worldbody/body/geom'):
                #     elem.set('density', str(self.density))
                for elem in tree.iterfind('default/geom'):
                    elem.set('density', str(self.density) + ' .1 .1')
                for elem in tree.iterfind('default/geom'):
                    elem.set('friction', str(self.friction) + ' .1 .1')

        return super(RandomNormalWalker2dEvaluate, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalWalker2dEvaluate, self).parameters
        parameters.update(
            {'density': self.density, 'friction': self.friction, })


# ============== Half Cheetah =================
class ModifiableRoboschoolHalfCheetah(RoboschoolHalfCheetah, RoboschoolTrackDistSuccessMixin):

    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 500
    EXTREME_UPPER_DENSITY = 1500

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1

    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 2.25

    RANDOM_LOWER_POWER = 0.7
    RANDOM_UPPER_POWER = 1.1
    EXTREME_LOWER_POWER = 0.5
    EXTREME_UPPER_POWER = 1.3

    def _reset(self, new=True):
        return super(ModifiableRoboschoolHalfCheetah, self)._reset()

    @property
    def parameters(self):
        return {'id': self.spec.id, }


class StrongHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(
            self, 'half_cheetah.xml', 'torso', action_dim=6, obs_dim=26, power=1.3)

    @property
    def parameters(self):
        parameters = super(StrongHalfCheetah, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class WeakHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(
            self, 'half_cheetah.xml', 'torso', action_dim=6, obs_dim=26, power=0.5)

    @property
    def parameters(self):
        parameters = super(WeakHalfCheetah, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class RandomStrongHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def randomize_power(self):
        self.power = self.np_random.uniform(
            self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

    def _reset(self, new=True):
        if new:
            self.randomize_power()
        return super(RandomStrongHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomStrongHalfCheetah, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class RandomWeakHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def randomize_power(self):
        self.power = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_POWER, self.EXTREME_UPPER_POWER,
                                           self.RANDOM_LOWER_POWER, self.EXTREME_UPPER_POWER)

    def _reset(self, new=True):
        if new:
            self.randomize_power()
        return super(RandomWeakHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomWeakHalfCheetah, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class HeavyTorsoHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        self.density = 1500
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=6, obs_dim=26, power=0.9)

    @property
    def parameters(self):
        parameters = super(HeavyTorsoHalfCheetah, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class LightTorsoHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        self.density = 500
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=6, obs_dim=26, power=0.9)

    @property
    def parameters(self):
        parameters = super(LightTorsoHalfCheetah, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class RandomHeavyTorsoHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def randomize_mass(self):
        self.density = self.np_random.uniform(
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

    def _reset(self, new=True):
        if new:
            self.randomize_mass()
        return super(RandomHeavyTorsoHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomHeavyTorsoHalfCheetah, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class RandomLightTorsoHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def randomize_mass(self):
        self.density = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_DENSITY, self.EXTREME_UPPER_DENSITY,
                                             self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

    def _reset(self, new=True):
        if new:
            self.randomize_mass()
        return super(RandomLightTorsoHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomLightTorsoHalfCheetah, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class SlipperyJointsHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        self.friction = 0.2
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction)+' .1 .1')

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=6, obs_dim=26, power=0.9)

    @property
    def parameters(self):
        parameters = super(SlipperyJointsHalfCheetah, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RoughJointsHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        self.friction = 1.4
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction)+' .1 .1')

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=6, obs_dim=26, power=0.9)

    @property
    def parameters(self):
        parameters = super(RoughJointsHalfCheetah, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomRoughJointsHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def randomize_friction(self):
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomRoughJointsHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomRoughJointsHalfCheetah, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomSlipperyJointsHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def randomize_friction(self):
        self.friction = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION,
                                              self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomSlipperyJointsHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomSlipperyJointsHalfCheetah, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomNormalHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):

    def randomize_env(self):
        self.density = self.np_random.uniform(
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        self.power = self.np_random.uniform(
            self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalHalfCheetah, self).parameters
        parameters.update(
            {'power': self.power, 'density': self.density, 'friction': self.friction, })
        return parameters


class RandomNormalHalfcheetahEvaluate(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):

    def set_envparam(self, density, friction):
        self.density = density
        self.friction = friction

        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            # self.randomize_env()
            print('==============in reset ==========')
            print(self.density)
            print(self.friction)
            with self.modify_xml('half_cheetah.xml') as tree:
                for elem in tree.iterfind('worldbody/body/geom'):
                    elem.set('density', str(self.density))
                for elem in tree.iterfind('default/geom'):
                    elem.set('friction', str(self.friction) + ' .1 .1')

        return super(RandomNormalHalfcheetahEvaluate, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalHalfcheetahEvaluate, self).parameters
        parameters.update(
            {'density': self.density, 'friction': self.friction, })


class RandomExtremeHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):

    def randomize_env(self):
        '''
        # self.armature = self.np_random.uniform(0.2, 0.5)
        self.density = self.np_random.uniform(self.LOWER_DENSITY, self.UPPER_DENSITY)
        self.friction = self.np_random.uniform(self.LOWER_FRICTION, self.UPPER_FRICTION)
        self.power = self.np_random.uniform(self.LOWER_POWER, self.UPPER_POWER)
        '''

        self.density = uniform_exclude_inner(self.np_random.uniform,
                                             self.EXTREME_LOWER_DENSITY, self.EXTREME_UPPER_DENSITY,
                                             self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = uniform_exclude_inner(self.np_random.uniform,
                                              self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION,
                                              self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        self.power = uniform_exclude_inner(self.np_random.uniform,
                                           self.EXTREME_LOWER_POWER, self.EXTREME_UPPER_POWER,
                                           self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomExtremeHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomExtremeHalfCheetah, self).parameters
        parameters.update(
            {'power': self.power, 'density': self.density, 'friction': self.friction, })
        return parameters


# =========== Hopper ===============
class ModifiableRoboschoolHopper(RoboschoolHopper, RoboschoolTrackDistSuccessMixin):

    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 500
    EXTREME_UPPER_DENSITY = 1500

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1
    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 1.4

    RANDOM_LOWER_POWER = 0.6
    RANDOM_UPPER_POWER = 0.9
    EXTREME_LOWER_POWER = 0.4
    EXTREME_UPPER_POWER = 1.1

    def _reset(self, new=True):
        return super(ModifiableRoboschoolHopper, self)._reset()

    @property
    def parameters(self):
        return {'id': self.spec.id, }


class StrongHopper(ModifiableRoboschoolHopper):
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(
            self, 'hopper.xml', 'torso', action_dim=3, obs_dim=15, power=1.1)

    @property
    def parameters(self):
        parameters = super(StrongHopper, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class WeakHopper(ModifiableRoboschoolHopper):
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(
            self, 'hopper.xml', 'torso', action_dim=3, obs_dim=15, power=0.4)

    @property
    def parameters(self):
        parameters = super(WeakHopper, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class RandomStrongHopper(ModifiableRoboschoolHopper):
    def randomize_power(self):
        self.power = self.np_random.uniform(
            self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

    def _reset(self, new=True):
        if new:
            self.randomize_power()
        return super(RandomStrongHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomStrongHopper, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class RandomWeakHopper(ModifiableRoboschoolHopper):
    def randomize_power(self):
        self.power = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_POWER, self.EXTREME_UPPER_POWER,
                                           self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

    def _reset(self, new=True):
        if new:
            self.randomize_power()
        return super(RandomWeakHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomWeakHopper, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class HeavyTorsoHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def __init__(self):
        self.density = 1500
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=3, obs_dim=15, power=0.75)

    @property
    def parameters(self):
        parameters = super(HeavyTorsoHopper, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class LightTorsoHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def __init__(self):
        self.density = 500
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=3, obs_dim=15, power=0.75)

    @property
    def parameters(self):
        parameters = super(LightTorsoHopper, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class RandomHeavyTorsoHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def randomize_mass(self):
        self.density = self.np_random.uniform(
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

    def _reset(self, new=True):
        if new:
            self.randomize_mass()
        return super(RandomHeavyTorsoHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomHeavyTorsoHopper, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class RandomLightTorsoHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def randomize_mass(self):
        self.density = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_DENSITY, self.EXTREME_UPPER_DENSITY,
                                             self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

    def _reset(self, new=True):
        if new:
            self.randomize_mass()
        return super(RandomLightTorsoHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomLightTorsoHopper, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class SlipperyJointsHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def __init__(self):
        self.friction = 0.2
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction)+' .1 .1')

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=3, obs_dim=15, power=0.75)

    @property
    def parameters(self):
        parameters = super(SlipperyJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RoughJointsHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def __init__(self):
        self.friction = 1.4
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction)+' .1 .1')

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=3, obs_dim=15, power=0.75)

    @property
    def parameters(self):
        parameters = super(RoughJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomRoughJointsHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def randomize_friction(self):
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomRoughJointsHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomRoughJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomSlipperyJointsHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def randomize_friction(self):
        self.friction = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION,
                                              self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomSlipperyJointsHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomSlipperyJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomSlipperyJointsHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def randomize_friction(self):
        self.friction = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION,
                                              self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomSlipperyJointsHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomSlipperyJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomExtremeSlipperyJointsHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def randomize_friction(self):
        self.friction = self.np_random.uniform(
            self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION)

        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomExtremeSlipperyJointsHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomExtremeSlipperyJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomExtremeSlipperyJointsHopperEval(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def set_envparam(self, friction):
        self.friction = friction

        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            with self.modify_xml('hopper.xml') as tree:
                for elem in tree.iterfind('default/geom'):
                    elem.set('friction', str(self.friction) + ' .1 .1')

        return super(RandomExtremeSlipperyJointsHopperEval, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(
            RandomExtremeSlipperyJointsHopperEval, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomNormalHopperEvaluate(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):

    def set_envparam(self, density, friction):
        self.density = density
        self.friction = friction

        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            # self.randomize_env()
            print('==============in reset ==========')
            print(self.density)
            print(self.friction)
            with self.modify_xml('hopper.xml') as tree:
                for elem in tree.iterfind('worldbody/body/geom'):
                    elem.set('density', str(self.density))
                for elem in tree.iterfind('default/geom'):
                    elem.set('friction', str(self.friction) + ' .1 .1')

        return super(RandomNormalHopperEvaluate, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalHopperEvaluate, self).parameters
        parameters.update(
            {'power': self.power, 'density': self.density, 'friction': self.friction, })
        return parameters


class RandomNormalHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):

    def randomize_env(self):
        self.density = self.np_random.uniform(
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        self.power = self.np_random.uniform(
            self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalHopper, self).parameters
        parameters.update(
            {'power': self.power, 'density': self.density, 'friction': self.friction, })
        return parameters


class RandomExtremeHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):

    def randomize_env(self):
        '''
        self.density = self.np_random.uniform(self.LOWER_DENSITY, self.UPPER_DENSITY)
        self.friction = self.np_random.uniform(self.LOWER_FRICTION, self.UPPER_FRICTION)
        self.power = self.np_random.uniform(self.LOWER_POWER, self.UPPER_POWER)
        '''

        self.density = uniform_exclude_inner(self.np_random.uniform,
                                             self.EXTREME_LOWER_DENSITY, self.EXTREME_UPPER_DENSITY,
                                             self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = uniform_exclude_inner(self.np_random.uniform,
                                              self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION,
                                              self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        self.power = uniform_exclude_inner(self.np_random.uniform,
                                           self.EXTREME_LOWER_POWER, self.EXTREME_UPPER_POWER,
                                           self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomExtremeHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomExtremeHopper, self).parameters
        parameters.update(
            {'power': self.power, 'density': self.density, 'friction': self.friction, })
        return parameters
