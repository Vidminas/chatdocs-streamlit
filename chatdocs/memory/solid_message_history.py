import requests

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage


class CssAccount:
    def __init__(
        self,
        css_base_url: str,
        name: str,
        email: str,
        password: str,
        web_id: str,
        pod_base_url: str,
    ) -> None:
        self.css_base_url = css_base_url
        self.name = name
        self.email = email
        self.password = password
        self.web_id = web_id
        self.pod_base_url = pod_base_url


class ClientCredentials:
    def __init__(self, client_id: str, client_secret: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret


def create_css_account(
    css_base_url: str, name: str, email: str, password: str
) -> CssAccount:
    register_endpoint = f"{css_base_url}/idp/register/"

    res = requests.post(
        register_endpoint,
        json={
            "createWebId": "on",
            "webId": "",
            "register": "on",
            "createPod": "on",
            "podName": name,
            "email": email,
            "password": password,
            "confirmPassword": password,
        },
        timeout=5000,
    )

    if not res.ok:
        raise Exception(f"Could not create account: {res.status_code} {res.text}")

    data = res.json()
    account = CssAccount(
        css_base_url=css_base_url,
        name=name,
        email=email,
        password=password,
        web_id=data["webId"],
        pod_base_url=data["podBaseUrl"],
    )
    return account


def get_client_credentials(account: CssAccount) -> ClientCredentials:
    credentials_endpoint = f"{account.css_base_url}/idp/credentials/"

    res = requests.post(
        credentials_endpoint,
        json={
            "name": "chatdocs-client-credentials",
            "email": account.email,
            "password": account.password,
        },
        timeout=5000,
    )

    if not res.ok:
        raise Exception(
            f"Could not create client credentials: {res.status_code} {res.text}"
        )

    data = res.json()
    return ClientCredentials(client_id=data["id"], client_secret=data["secret"])

def get_item_name(url) -> str:
    if url[-1] == '/':
        url = url[:-1]

    if url.count('/') == 2:  # is base url, no item name
        return ''

    i = url.rindex('/')
    return url[i + 1:]

class SolidChatMessageHistory(BaseChatMessageHistory):
    """
    Chat message history that stores messages in a Solid pod.

    Args:
        solid_server_url: A Community Solid Server base url.
    """

    def __init__(self, solid_server_url, account):
        try:
            from solid_client_credentials import SolidClientCredentialsAuth, DpopTokenProvider
        except ImportError as e:
            raise ImportError(
                "Unable to import solid_client_credentials, please run `pip install SolidClientCredentials`."
            ) from e
        try:
            from rdflib import Graph
        except ImportError as e:
            raise ImportError(
                "Unable to import rdflib, please run `pip install rdflib`."
            ) from e

        self.account = account
        client_credentials = get_client_credentials(account)
        token_provider = DpopTokenProvider(
            issuer_url=solid_server_url,
            client_id=client_credentials.client_id,
            client_secret=client_credentials.client_secret
        )
        self.session = requests.Session()
        self.session.auth = SolidClientCredentialsAuth(token_provider)
        self.graph = Graph()

    def is_item_available(self, url) -> bool:
        try:
            res = self.session.head(url, allow_redirects=True)
            return res.ok
        except requests.exceptions.ConnectionError:
            return False
        
    def create_item(self, url: str) -> bool:
        res = self.session.put(url,
                          data=None,
                          headers={
                              "Accept": 'text/turtle',
                              "If-None-Match": "*",
                              'Link': '<http://www.w3.org/ns/ldp#BasicContainer>; rel="type"'
                                      if url.endswith("/") else
                                      '<http://www.w3.org/ns/ldp#Resource>; rel="type"',
                              'Slug': get_item_name(url),
                              'Content-Type': 'text/turtle',
                          })
        return res.ok

    @property
    def messages(self) -> list[BaseMessage]:
        """Retrieve the current list of messages"""
        if not self.is_item_available(f"{self.account.pod_base_url}private/"):
            self.create_item(f"{self.account.pod_base_url}private/")
        if not self.is_item_available(f"{self.account.pod_base_url}private/chatdocs.ttl"):
            self.create_item(f"{self.account.pod_base_url}private/chatdocs.ttl")

        res = self.session.get(f"{self.account.pod_base_url}private/chatdocs.ttl")
        if not res.ok:
            print("getting messages failed", res.text)
            msgs = []
        else:
            from rdflib.namespace import PROF, RDF
            from rdflib.collection import Collection

            self.graph.parse(data=res.text, publicID=f"{self.account.pod_base_url}private/chatdocs.ttl")
            list_node = self.graph.value(predicate=RDF.type, object=RDF.List)
            if list_node is None:
                return []
            
            rdf_list = Collection(self.graph, list_node)
            msgs = [BaseMessage(
                content=self.graph.value(subject=msg, predicate=PROF.hasResource).toPython(),
                type=self.graph.value(subject=msg, predicate=PROF.hasRole).toPython()
                ) for msg in rdf_list]
        return msgs

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the session memory"""
        # https://solidproject.org/TR/protocol#n3-patch seems to be broken with Community Solid Server
        # https://www.w3.org/TR/sparql11-update/ works
        from rdflib import Graph
        from rdflib.term import Node, BNode, URIRef, Literal
        from rdflib.namespace import RDF, PROF, XSD
        from rdflib.collection import Collection

        update_graph = Graph()

        msg = BNode()
        update_graph.add((msg, RDF.type, PROF.ResourceDescriptor))
        update_graph.add((msg, PROF.hasResource, Literal(message.content, datatype=XSD.string)))
        update_graph.add((msg, PROF.hasRole, Literal(message.type, datatype=XSD.string)))

        list_node = self.graph.value(predicate=RDF.type, object=RDF.List)
        if list_node is None:
            msgs_node = URIRef(f"{self.account.pod_base_url}private/chatdocs.ttl#messages")
            update_graph.add((msgs_node, RDF.type, RDF.List))

            msgs = Collection(update_graph, msgs_node)
            msgs.append(msg)

            triples = "\n".join([
                f"{subject.n3()} {predicate.n3()} {object.n3()} ."
                for subject, predicate, object in update_graph
            ])
            sparql = f"INSERT DATA {{{triples}}}"
        else:
            new_item = BNode()
            update_graph.add((new_item, RDF.first, msg))
            update_graph.add((new_item, RDF.rest, RDF.nil))

            triples = "\n".join([
                f"{subject.n3()} {predicate.n3()} {object.n3()} ."
                for subject, predicate, object in update_graph
            ])
            sparql = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                DELETE {{ ?end rdf:rest rdf:nil }}
                INSERT {{ ?end rdf:rest {new_item.n3()} .\n
                          {triples} }}
                WHERE {{ ?end  rdf:rest  rdf:nil }}
            """

        # Update remote copy
        self.session.patch(
            url=f"{self.account.pod_base_url}private/chatdocs.ttl",
            data=sparql.encode("utf-8"),
            headers={
                "Content-Type": "application/sparql-update",
            }
        )
        # Update local copy
        self.graph.update(sparql)

    def clear(self) -> None:
        """Clear session memory"""
        from rdflib import Graph

        self.session.delete(f"{self.account.pod_base_url}private/chatdocs.ttl")
        self.graph = Graph()
