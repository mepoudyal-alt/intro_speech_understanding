import unittest, homework3

class Test(unittest.TestCase):
    def test_wrods2characters(self):
        words = [ 'hello', 1.234, True ]
        correct = ['h','e','l','l','o','1','.','2','3','4','T','r','u','e']
        characters = homework3.words2characters(words)
        self.assertEqual(len(characters), len(correct),
                         'characters has %d elements, should have %d'%(len(characters),len(correct)))
        for i in range(len(characters)):
            self.assertEqual(characters[i],correct[i],
                             'characters[%d] is "%s", should be "%s"'%(i,characters[i],correct[i]))
                         
        
suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
result = unittest.TextTestRunner().run(suite)

n_success = result.testsRun - len(result.errors) - len(result.failures)
print('%d successes out of %d tests run'%(n_success, result.testsRun))
print('Score: %d%%'%(int(100*(n_success/result.testsRun))))
