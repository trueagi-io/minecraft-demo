from tagilmo import VereyaPython
import time
import sys
import unittest
import logging

logger = logging.getLogger(__name__)

def run_tests(test_files):
    # create a test suite from the list of test files
    for test_file in test_files:
        suite = unittest.TestSuite()
        # load the tests from the file
        tests = unittest.defaultTestLoader.loadTestsFromName(test_file)
        # create a test runner and run the tests
        runner = unittest.TextTestRunner()
        logger.info('running ' + test_file)
        result = runner.run(tests)
        if not result.wasSuccessful():
            logger.info('suite '+ test_file + ' failed')
            return result
        logger.info('sleeping')
        time.sleep(5)
    return result


def main():
    VereyaPython.setupLogger()
    test_files = ['test_motion_vereya', 'test_craft', 
                  'test_inventory', 'test_quit', 
                  'test_observation', 'test_placement', 'test_image',
                  'test_consistency']
    res = run_tests(test_files)
    if not res.wasSuccessful():
        sys.exit(1)


if __name__ == '__main__':
    main()
