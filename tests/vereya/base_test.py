"""
base test case class
"""

import unittest
import logging

logger = logging.getLogger(__name__)


class BaseTest(unittest.TestCase):

    def getTestName(self):
        return self.id().split('.')[-1]

    def setUp(self):
        # log current test name
        logger.info("start test " + self.getTestName())
        super().setUp()

    def tearDown(self):
        # log current test name
        logger.info("end test " + self.getTestName())
        super().tearDown()

    # wrap asserts to log them
    def assertEqual(self, first, second, msg=None):
        # log if not equal
        if first != second:
            logger.info(f"assertEqual failed in {self.getTestName()}" + str(first) + " != " + str(second) + str(msg))
        super().assertEqual(first, second, msg)

    def assertNotEqual(self, first, second, msg=None):
        # log if equal
        if first == second:
            logger.info(f"assertNotEqual failed in {self.getTestName()}" + str(first) + " == " + str(second) + str(msg))
        super().assertNotEqual(first, second, msg)

    def assertTrue(self, expr, msg=None):
        # log if not true
        if not expr:
            logger.info(f"assertTrue failed in {self.getTestName()}" + str(expr) + str(msg))
        super().assertTrue(expr, msg)

    def assertFalse(self, expr, msg=None):
        # log if not false
        if expr:
            logger.info(f"assertFalse failed in {getTestName()}" + str(expr) + str(msg))
        super().assertFalse(expr, msg)

    def assertGreater(self, a, b, msg=None):
        # log if not greater
        if a <= b:
            logger.info(f"assertGreater failed in {getTestName()}" + str(a) + " <= " + str(b) + str(msg))
        super().assertGreater(a, b, msg)
