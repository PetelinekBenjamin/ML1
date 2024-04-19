import os
import subprocess
import unittest

class TestCodeQuality(unittest.TestCase):
    def test_code_quality(self):
        # preverimo kakovost kode
        result = subprocess.run(['flake8', '--exclude', '.git,__pycache__,venv', '--count', '--select=E9,F63,F7,F82', '--show-source', '--statistics'], stdout=subprocess.PIPE)

        # Če je število najdenih napak večje od 0, test pade
        self.assertEqual(result.returncode, 0, f"Koda ni v redu. Povzetek napak: {result.stdout.decode('utf-8')}")


if __name__ == '__main__':
    unittest.main()
