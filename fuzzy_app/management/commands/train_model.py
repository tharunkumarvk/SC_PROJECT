"""
Management command to train the Random Forest model and generate the dataset.

Usage:
    python manage.py train_model
    python manage.py train_model --samples 300
"""

import os
from django.core.management.base import BaseCommand
from fuzzy_app.ml_model import train_model
from fuzzy_app.tropical_diseases import generate_dataset, save_dataset_csv


class Command(BaseCommand):
    help = 'Train the Random Forest model for tropical disease prediction'

    def add_arguments(self, parser):
        parser.add_argument(
            '--samples', type=int, default=250,
            help='Number of samples per disease (default: 250)'
        )
        parser.add_argument(
            '--save-csv', action='store_true',
            help='Also save the generated dataset as CSV'
        )

    def handle(self, *args, **options):
        samples = options['samples']

        self.stdout.write(self.style.NOTICE(
            f'\nTraining Random Forest with {samples} samples per disease '
            f'({samples * 10} total, 10 diseases)...\n'
        ))

        # Optionally save dataset
        if options['save_csv']:
            data, keys = generate_dataset(samples_per_disease=samples)
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                'tropical_disease_dataset.csv'
            )
            save_dataset_csv(csv_path, data, keys)
            self.stdout.write(self.style.SUCCESS(f'Dataset saved to {csv_path}'))

        # Train model
        info = train_model(samples_per_disease=samples, verbose=True)

        self.stdout.write(self.style.SUCCESS(
            f'\nModel trained successfully!'
            f'\n  Cross-validation accuracy: {info["cv_accuracy"]:.4f} '
            f'(+/- {info["cv_std"]:.4f})'
        ))
