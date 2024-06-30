"""
ASGI config for propounding_first_artificial_intelligence_approach.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'propounding_first_artificial_intelligence_approach.settings')

application = get_asgi_application()
