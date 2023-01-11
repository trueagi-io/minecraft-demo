from tagilmo import VereyaPython
import time
import sys
import unittest


def run_tests(test_files):
    # create a test suite from the list of test files
    for test_file in test_files:
        suite = unittest.TestSuite()

        print('running ' + test_file)
        # load the tests from the file
        tests = unittest.defaultTestLoader.loadTestsFromName(test_file)
        # add the tests to the suite
        suite.addTests(tests)
    # create a test runner and run the tests
        runner = unittest.TextTestRunner()
        result = runner.run(suite)
        if not result.wasSuccessful():
            return result
        print('sleeping')
        time.sleep(5)
    return result


def main():
    VereyaPython.setupLogger()
    test_files = ['test_craft', 'test_inventory', 'test_quit', 'test_observation' ]
    res = run_tests(test_files)
    if not res.wasSuccessful():
        sys.exit(1)


if __name__ == '__main__':
    main()
