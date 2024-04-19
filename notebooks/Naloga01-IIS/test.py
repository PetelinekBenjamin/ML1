import os
import subprocess
import unittest

class TestCodeQuality(unittest.TestCase):
    def test_code_quality(self):
        # Preverimo kakovost kode
        result = subprocess.run(['flake8', '--exclude', '.git,__pycache__,venv', '--count', '--select=E9,F63,F7,F82', '--show-source', '--statistics'], stdout=subprocess.PIPE)

        # Preverimo, ali je test uspešen (vrne izhodno kodo 0)
        self.assertEqual(result.returncode, 1, f"Koda ni v redu. Povzetek napak: {result.stdout.decode('utf-8')}")
        


if __name__ == '__main__':
    unittest.main()
