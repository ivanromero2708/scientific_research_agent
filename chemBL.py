import streamlit as st
import requests
import pandas as pd

def get_chembl_data(query_type, query):
    """Fetches data from ChEMBL API based on query type and value."""
    base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"  # Ensure JSON format
    if query_type == "name":
        url = f"{base_url}?pref_name__icontains={query}&format=json"
    elif query_type == "cas":
        url = f"{base_url}?molecule_synonyms__synonyms__iexact={query}&format=json"
    elif query_type == "smiles":
        url = f"{base_url}?substructure={query}&format=json"
    else:
        return None

    try:
        response = requests.get(url, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        return response.json() if response.text.strip() else None
    except requests.exceptions.RequestException as e:
        st.error(f"Error al realizar la solicitud al API: {e}")
        return None

def get_drug_data(chembl_id):
    """Fetches drug data (market status, mechanism of action, indications) from ChEMBL."""
    try:
        # Fetch approved drugs information
        drug_url = f"https://www.ebi.ac.uk/chembl/api/data/drug?molecule_chembl_id={chembl_id}&format=json"
        mechanism_url = f"https://www.ebi.ac.uk/chembl/api/data/mechanism?molecule_chembl_id={chembl_id}&format=json"
        indication_url = f"https://www.ebi.ac.uk/chembl/api/data/drug_indication?molecule_chembl_id={chembl_id}&format=json"

        drug_response = requests.get(drug_url, headers={"Content-Type": "application/json"})
        mechanism_response = requests.get(mechanism_url, headers={"Content-Type": "application/json"})
        indication_response = requests.get(indication_url, headers={"Content-Type": "application/json"})

        drug_data = drug_response.json() if drug_response.status_code == 200 else None
        mechanism_data = mechanism_response.json() if mechanism_response.status_code == 200 else None
        indication_data = indication_response.json() if indication_response.status_code == 200 else None

        return drug_data, mechanism_data, indication_data
    except requests.exceptions.RequestException as e:
        st.error(f"Error al realizar la solicitud al API para datos de drogas: {e}")
        return None, None, None

def generate_report(chembl_id, molecular_properties, drug_data, mechanism_data, indication_data):
    """Generates a nicely formatted report for the results."""
    report = []

    report.append(f"# Reporte de Caracterización Fisicoquímica del API\n")
    report.append(f"**ChEMBL ID:** {chembl_id}\n")

    report.append(f"## Propiedades Moleculares\n")
    for key, value in molecular_properties.items():
        report.append(f"- **{key.replace('_', ' ').title()}:** {value}\n")

    if drug_data and "drugs" in drug_data:
        report.append("## Drogas en el Mercado\n")
        for drug in drug_data["drugs"]:
            report.append(f"- **Nombre Comercial:** {drug.get('trade_name', 'N/A')}\n")
            report.append(f"  - **Fabricante:** {', '.join(drug.get('applicants', ['N/A']))}\n")
            report.append(f"  - **ATC Classification:** {', '.join([atc.get('description', 'N/A') for atc in drug.get('atc_classification', [])])}\n")
            report.append(f"  - **Indicación:** {drug.get('indication_class', 'N/A')}\n")
            report.append(f"  - **Estado de Aprobación:** {drug.get('approval_status', 'N/A')}\n")

    if mechanism_data and "mechanisms" in mechanism_data:
        report.append("## Mecanismos de Acción\n")
        for mechanism in mechanism_data["mechanisms"]:
            report.append(f"- **Mecanismo de Acción:** {mechanism.get('mechanism_of_action', 'N/A')}\n")
            report.append(f"  - **Objetivo:** {mechanism.get('target_name', 'N/A')}\n")

    if indication_data and "drug_indications" in indication_data:
        report.append("## Indicaciones Terapéuticas\n")
        for indication in indication_data["drug_indications"]:
            report.append(f"- **Término EFO:** {indication.get('efo_term', 'N/A')}\n")
            report.append(f"  - **MESH Heading:** {indication.get('mesh_heading', 'N/A')}\n")
            if "indication_refs" in indication:
                refs = [f"[{ref.get('ref_id', 'N/A')}]({ref.get('ref_url', '#')})" for ref in indication["indication_refs"]]
                report.append(f"  - **Referencias:** {', '.join(refs)}\n")

    return "\n".join(report)

# Streamlit UI
st.set_page_config(page_title="Caracterización Fisicoquímica del API", layout="wide")
st.title("Caracterización Fisicoquímica del API")
st.sidebar.title("Estabilidad Ab Initio")
st.sidebar.markdown("**Caracterización del API**")

# Sidebar Menu
menu_items = [
    "Caracterización del API",
    "Sistema Experto de Predicción de Incompatibilidad",
    "Agente de Investigación de Literatura",
    "Generación de Reportes y Protocolos",
    "Pruebas de Estrés",
    "Asociar Carpeta SharePoint/Elara",
    "Checklists de Documentación"
]
menu_choice = st.sidebar.radio("Navegación", menu_items)

# Main Section: Search Form
if menu_choice == "Caracterización del API":
    st.subheader("Búsqueda de Compuesto")

    # Input for compound search
    search_option = st.selectbox(
        "Buscar por:",
        ("Nombre", "Número CAS", "Estructura SMILES")
    )

    query = st.text_input("Ingrese el término de búsqueda:", placeholder="Ej: aspirina, 50-78-2, o CC(=O)OC1=CC=CC=C1C(=O)O")

    if st.button("Buscar"):
        if search_option == "Nombre":
            results = get_chembl_data("name", query)
        elif search_option == "Número CAS":
            results = get_chembl_data("cas", query)
        elif search_option == "Estructura SMILES":
            results = get_chembl_data("smiles", query)
        else:
            results = None

        if results and "molecules" in results:
            st.success("Resultados encontrados:")

            # Extract relevant data
            molecule = results["molecules"][0]  # Assume only one result
            chembl_id = molecule.get("molecule_chembl_id")
            molecular_properties = molecule.get("molecule_properties", {})

            # Fetch and display drug data
            drug_data, mechanism_data, indication_data = get_drug_data(chembl_id)

            # Generate and display the report
            report = generate_report(chembl_id, molecular_properties, drug_data, mechanism_data, indication_data)
            st.markdown(report, unsafe_allow_html=True)

        else:
            st.error("No se encontraron resultados para el término de búsqueda.")
