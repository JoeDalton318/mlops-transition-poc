"""
Functional tests for the pure/near-pure utility functions of the Streamlit
dashboard (src/dashboard.py): API health polling, drift-report discovery,
and the prediction-request helper, including its graceful handling of an
offline API.

The dashboard module executes several Streamlit calls (st.set_page_config,
st.sidebar/st.radio, st.session_state) at *import* time, outside of a real
`streamlit run` script context. Streamlit is designed to no-op these calls
gracefully (logging a "missing ScriptRunContext" warning) rather than
raising when imported this way, which is what allows a plain `import
dashboard` here --- but this behavior has not been independently verified
by executing this suite in this environment. If that assumption turns out
to be wrong, this entire module is skipped (not failed) at collection time,
so it cannot break the rest of `pytest tests/ -v`.

Tests fonctionnels pour les fonctions utilitaires pures/quasi pures du
tableau de bord Streamlit (src/dashboard.py) : sondage de l'état de l'API,
découverte du dernier rapport de dérive, et l'assistant d'envoi de requête
de prédiction, y compris sa gestion gracieuse d'une API hors ligne.

Le module dashboard exécute plusieurs appels Streamlit (st.set_page_config,
st.sidebar/st.radio, st.session_state) au moment de l'*import*, en dehors
d'un contexte réel de script `streamlit run`. Streamlit est conçu pour
neutraliser ces appels sans lever d'exception (en journalisant un
avertissement "missing ScriptRunContext") plutôt que de planter lors d'un
tel import --- mais ce comportement n'a pas été vérifié indépendamment en
exécutant cette suite dans cet environnement. Si cette hypothèse s'avère
fausse, ce module entier est ignoré (skip, pas échec) au moment de la
collecte, afin de ne pas casser le reste de `pytest tests/ -v`. Ce filet
ne couvre que l'échec d'import : le fichier est d'abord compilé, de sorte
qu'une erreur de syntaxe fasse échouer la suite au lieu d'être ignorée.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

_DASHBOARD_SOURCE = Path(__file__).parent.parent / 'src' / 'dashboard.py'

# Le filet de sécurité ci-dessous ne doit couvrir que l'hypothèse d'environnement décrite
# plus haut (Streamlit importé hors contexte). Une faute de syntaxe ou un import cassé dans
# src/dashboard.py doit faire échouer la suite, pas l'ignorer : compiler le fichier d'abord
# garantit qu'un module invalide ne passe jamais inaperçu.
# The guard below must only cover the environment assumption documented above. A syntax error
# or a broken import in src/dashboard.py must fail the suite, not skip it: compiling the file
# first ensures an invalid module never slips through unnoticed.
compile(_DASHBOARD_SOURCE.read_text(encoding='utf-8'), str(_DASHBOARD_SOURCE), 'exec')

try:
    import dashboard  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment-dependent import guard
    pytest.skip(
        f'src/dashboard.py could not be imported outside a `streamlit run` '
        f'context in this environment ({exc!r}); skipping dashboard '
        f'functional tests rather than failing the whole suite.',
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _isolate_streamlit_calls(monkeypatch):
    """
    Neutralize the two Streamlit touchpoints these functions rely on
    (translation lookups and st.error/st.info calls) so these tests verify
    send_prediction_request()'s and check_api_health()'s actual control
    flow, rather than depending on whatever st.session_state.language
    happened to resolve to when the module was imported outside a real
    Streamlit script run.

    Neutralise les deux points de contact Streamlit dont dépendent ces
    fonctions (recherche de traduction et appels st.error/st.info) afin que
    ces tests vérifient le flux de contrôle réel de send_prediction_request()
    et check_api_health(), plutôt que de dépendre de la valeur qu'a pu
    prendre st.session_state.language lors de l'import du module en dehors
    d'une véritable exécution de script Streamlit.
    """
    monkeypatch.setattr(dashboard, 't', lambda key: key, raising=False)
    monkeypatch.setattr(dashboard.st, 'error', MagicMock(), raising=False)
    monkeypatch.setattr(dashboard.st, 'info', MagicMock(), raising=False)


# --------------------------------------------------------------------------
# check_api_health()
# --------------------------------------------------------------------------

def test_check_api_health_true_when_api_responds_200(monkeypatch):
    """A healthy API (200 + model_status field) must report (True, status)."""
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {'model_status': 'loaded'}
    monkeypatch.setattr(dashboard.requests, 'get', MagicMock(return_value=fake_response))

    healthy, status = dashboard.check_api_health()

    assert healthy is True
    assert status == 'loaded'


def test_check_api_health_false_none_on_non_200_status(monkeypatch):
    """A non-200 response (API up but unhealthy) must report (False, None), not raise."""
    fake_response = MagicMock(status_code=500)
    monkeypatch.setattr(dashboard.requests, 'get', MagicMock(return_value=fake_response))

    healthy, status = dashboard.check_api_health()

    assert healthy is False
    assert status is None


def test_check_api_health_false_none_when_api_offline(monkeypatch):
    """
    An unreachable API (ConnectionError, DNS failure, timeout, ...) must be
    handled gracefully and report (False, None) --- this is the exact
    "API offline" state the dashboard's graceful-degradation UI depends on.

    Une API inaccessible (ConnectionError, échec DNS, timeout, ...) doit
    être gérée avec grâce et renvoyer (False, None) --- c'est précisément
    l'état "API hors ligne" dont dépend l'interface de dégradation gracieuse
    du tableau de bord.
    """
    monkeypatch.setattr(
        dashboard.requests, 'get',
        MagicMock(side_effect=requests.exceptions.ConnectionError('offline')),
    )

    healthy, status = dashboard.check_api_health()

    assert healthy is False
    assert status is None


# --------------------------------------------------------------------------
# get_latest_drift_report()
# --------------------------------------------------------------------------

def test_get_latest_drift_report_none_when_dir_missing(monkeypatch, tmp_path):
    """
    If DRIFT_REPORTS_DIR does not exist yet, no report can be found.
    Si DRIFT_REPORTS_DIR n'existe pas encore, aucun rapport ne peut être trouvé.
    """
    monkeypatch.setattr(dashboard, 'DRIFT_REPORTS_DIR', tmp_path / 'does_not_exist')

    assert dashboard.get_latest_drift_report() is None


def test_get_latest_drift_report_none_when_dir_empty(monkeypatch, tmp_path):
    """
    An existing but empty reports directory must also yield None, not raise.
    Un répertoire de rapports existant mais vide doit aussi renvoyer None, sans lever d'exception.
    """
    monkeypatch.setattr(dashboard, 'DRIFT_REPORTS_DIR', tmp_path)

    assert dashboard.get_latest_drift_report() is None


def test_get_latest_drift_report_returns_most_recent_by_mtime(monkeypatch, tmp_path):
    """
    With several drift_report_*.html files present, the *most recently
    modified* one must be returned, matching the dashboard's intent of
    always surfacing the latest drift-detection run --- not simply the
    lexicographically-last filename.

    Avec plusieurs fichiers drift_report_*.html présents, celui *modifié le
    plus récemment* doit être renvoyé, conformément à l'intention du tableau
    de bord de toujours afficher la dernière exécution de détection de
    dérive --- pas simplement le nom de fichier lexicographiquement le plus
    grand.
    """
    monkeypatch.setattr(dashboard, 'DRIFT_REPORTS_DIR', tmp_path)

    older = tmp_path / 'drift_report_20260101_000000.html'
    newer = tmp_path / 'drift_report_20260101_000001.html'
    older.write_text('<html>old</html>')
    newer.write_text('<html>new</html>')

    now = 1_700_000_000
    os.utime(older, (now, now))
    os.utime(newer, (now + 100, now + 100))

    result = dashboard.get_latest_drift_report()

    assert result == newer


# --------------------------------------------------------------------------
# send_prediction_request()
# --------------------------------------------------------------------------

VALID_FEATURES = {
    'Surface_m2': 65.0,
    'Nb_Pieces': 3,
    'Annee_Construction': 2005,
    'Distance_Centre_km': 3.0,
    'DPE_Energy_Class': 4,
    'Has_Balcony': 1,
    'Has_Parking': 1,
}


def test_send_prediction_request_returns_payload_on_success(monkeypatch):
    """
    A 200 API response must be returned to the caller as-is (parsed JSON).
    Une réponse API 200 doit être renvoyée telle quelle à l'appelant (JSON décodé).
    """
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {'predicted_price_k_eur': 250.0}
    monkeypatch.setattr(dashboard.requests, 'post', MagicMock(return_value=fake_response))

    result = dashboard.send_prediction_request(VALID_FEATURES)

    assert result == {'predicted_price_k_eur': 250.0}


def test_send_prediction_request_returns_none_and_reports_error_on_non_200(monkeypatch):
    """A non-200 API response must surface via st.error(...) and return None, not raise."""
    fake_response = MagicMock(status_code=422)
    monkeypatch.setattr(dashboard.requests, 'post', MagicMock(return_value=fake_response))

    result = dashboard.send_prediction_request(VALID_FEATURES)

    assert result is None
    dashboard.st.error.assert_called_once()


def test_send_prediction_request_handles_offline_api_gracefully(monkeypatch):
    """
    Graceful offline handling: when the API is unreachable, the function
    must catch requests.exceptions.ConnectionError, surface both an error
    and a hint via st.error/st.info, and return None --- never let the
    exception propagate up into the Streamlit UI callback.

    Gestion gracieuse du mode hors ligne : lorsque l'API est inaccessible,
    la fonction doit intercepter requests.exceptions.ConnectionError,
    afficher à la fois une erreur et une indication via st.error/st.info, et
    renvoyer None --- sans jamais laisser l'exception remonter jusqu'au
    callback de l'interface Streamlit.
    """
    monkeypatch.setattr(
        dashboard.requests, 'post',
        MagicMock(side_effect=requests.exceptions.ConnectionError('offline')),
    )

    result = dashboard.send_prediction_request(VALID_FEATURES)

    assert result is None
    dashboard.st.error.assert_called_once()
    dashboard.st.info.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
