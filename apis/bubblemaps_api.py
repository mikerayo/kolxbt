"""
Bubblemaps API Integration - Obtiene datos de distribución de tokens

API Docs: https://docs.bubblemaps.io/data/api
Endpoint: /maps/{chain}/{token_address}

Información que obtiene:
- Top 80 holders y sus porcentajes
- Clusters de wallets conectadas
- Decentralization score
- Supply en CEXs/DEXs
- Magic nodes (dev, insider, etc.)
"""

import aiohttp
import asyncio
from typing import Dict, List, Optional
from datetime import datetime


class BubblemapsAPI:
    """
    Cliente para la API de Bubblemaps
    """

    def __init__(self, api_key: str = None):
        """
        Inicializa el cliente de Bubblemaps

        Args:
            api_key: API key de Bubblemaps
        """
        self.api_key = api_key or "JZSt2a4s09aP0oRj6WrW"
        self.base_url = "https://api.bubblemaps.io"
        self.session = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def get_map_data(
        self,
        token_address: str,
        chain: str = "solana",
        use_magic_nodes: bool = False,
        return_nodes: bool = True,
        return_relationships: bool = False,
        return_clusters: bool = True,
        return_decentralization_score: bool = True
    ) -> Optional[Dict]:
        """
        Obtiene datos completos del mapa de un token

        Args:
            token_address: Contract address del token
            chain: Blockchain (solana, eth, bsc, etc.)
            use_magic_nodes: Incluir magic nodes (dev, CEX, etc.)
            return_nodes: Retornar nodos/holders
            return_relationships: Retornar relaciones
            return_clusters: Retornar clusters
            return_decentralization_score: Retornar score de descentralización

        Returns:
            Dict con datos del mapa o None si error
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with BubblemapsAPI()'")

        # Build query parameters
        params = {
            'use_magic_nodes': str(use_magic_nodes).lower(),
            'return_nodes': str(return_nodes).lower(),
            'return_relationships': str(return_relationships).lower(),
            'return_clusters': str(return_clusters).lower(),
            'return_decentralization_score': str(return_decentralization_score).lower()
        }

        url = f"{self.base_url}/maps/{chain}/{token_address}"
        headers = {
            'X-ApiKey': self.api_key
        }

        try:
            async with self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_map_data(data)
                elif response.status == 404:
                    print(f"[!] No holders found for token {token_address[:8]}...")
                    return None
                elif response.status == 400:
                    print(f"[!] Unsupported token or chain: {token_address[:8]}...")
                    return None
                else:
                    print(f"[!] Error {response.status}: {await response.text()}")
                    return None

        except asyncio.TimeoutError:
            print(f"[!] Timeout fetching Bubblemaps data for {token_address[:8]}...")
            return None
        except Exception as e:
            print(f"[!] Error fetching Bubblemaps data: {e}")
            return None

    def _parse_map_data(self, data: Dict) -> Dict:
        """
        Parsea y estructura los datos de Bubblemaps

        Args:
            data: Response cruda de Bubblemaps API

        Returns:
            Dict con datos procesados
        """
        result = {
            'token_address': None,
            'chain': None,
            'metadata': {},
            'holders': [],
            'clusters': [],
            'metrics': {}
        }

        # Metadata
        if 'metadata' in data:
            metadata = data['metadata']
            result['metadata'] = {
                'last_updated': metadata.get('dt_update'),
                'timestamp': metadata.get('ts_update'),
                'identified_supply': metadata.get('identified_supply', {})
            }

        # Top holders
        if 'nodes' in data and 'top_holders' in data['nodes']:
            holders = data['nodes']['top_holders']

            # Procesar holders
            for holder in holders[:80]:  # Top 80
                holder_info = {
                    'address': holder.get('address'),
                    'amount': holder.get('holder_data', {}).get('amount'),
                    'share': holder.get('holder_data', {}).get('share'),  # 0-1 format
                    'rank': holder.get('holder_data', {}).get('rank'),
                    'is_cex': holder.get('address_details', {}).get('is_cex', False),
                    'is_dex': holder.get('address_details', {}).get('is_dex', False),
                    'is_contract': holder.get('address_details', {}).get('is_contract', False),
                    'is_supernode': holder.get('address_details', {}).get('is_supernode', False),
                    'label': holder.get('address_details', {}).get('label'),
                    'entity_id': holder.get('address_details', {}).get('entity_id')
                }
                result['holders'].append(holder_info)

        # Clusters
        if 'clusters' in data:
            clusters = data['clusters']

            for cluster in clusters:
                cluster_info = {
                    'share': cluster.get('share'),  # 0-1 format
                    'amount': cluster.get('amount'),
                    'holder_count': cluster.get('holder_count'),
                    'holders': cluster.get('holders', [])
                }
                result['clusters'].append(cluster_info)

        # Decentralization score
        if 'decentralization_score' in data:
            result['metrics']['decentralization_score'] = data['decentralization_score']

        # Calculate additional metrics
        if result['holders']:
            result['metrics'] = self._calculate_metrics(result['holders'], result.get('clusters', []))

        return result

    def _calculate_metrics(self, holders: List[Dict], clusters: List[Dict]) -> Dict:
        """
        Calcula métricas adicionales desde los datos de holders

        Args:
            holders: Lista de holders con sus datos
            clusters: Lista de clusters

        Returns:
            Dict con métricas calculadas
        """
        metrics = {}

        if not holders:
            return metrics

        # Top holder percentage (rank #1)
        if holders:
            top_holder = holders[0]
            metrics['top1_percentage'] = top_holder.get('share', 0) * 100

            # Top 10 holders percentage
            top10_share = sum([h.get('share', 0) for h in holders[:10]])
            metrics['top10_percentage'] = top10_share * 100

            # Top 20 holders percentage
            top20_share = sum([h.get('share', 0) for h in holders[:20]])
            metrics['top20_percentage'] = top20_share * 100

            # Exclude CEXs and DEXs from concentration calculation
            retail_holders = [h for h in holders if not h.get('is_cex') and not h.get('is_dex')]
            if retail_holders:
                top10_retail = sum([h.get('share', 0) for h in retail_holders[:10]])
                metrics['top10_retail_percentage'] = top10_retail * 100

        # Contract holdings
        contract_holders = [h for h in holders if h.get('is_contract')]
        if contract_holders:
            contract_share = sum([h.get('share', 0) for h in contract_holders])
            metrics['contract_percentage'] = contract_share * 100
        else:
            metrics['contract_percentage'] = 0

        # CEX holdings
        cex_holders = [h for h in holders if h.get('is_cex')]
        if cex_holders:
            cex_share = sum([h.get('share', 0) for h in cex_holders])
            metrics['cex_percentage'] = cex_share * 100
        else:
            metrics['cex_percentage'] = 0

        # DEX holdings
        dex_holders = [h for h in holders if h.get('is_dex')]
        if dex_holders:
            dex_share = sum([h.get('share', 0) for h in dex_holders])
            metrics['dex_percentage'] = dex_share * 100
        else:
            metrics['dex_percentage'] = 0

        # Supernode count (whales with lots of activity)
        supernode_count = sum([1 for h in holders if h.get('is_supernode')])
        metrics['supernode_count'] = supernode_count

        # Total holder count (from top 80)
        metrics['holder_count'] = len(holders)

        # Largest cluster percentage
        if clusters:
            largest_cluster = max(clusters, key=lambda x: x.get('share', 0))
            metrics['largest_cluster_percentage'] = largest_cluster.get('share', 0) * 100
            metrics['cluster_count'] = len(clusters)
        else:
            metrics['largest_cluster_percentage'] = 0
            metrics['cluster_count'] = 0

        # Gini coefficient approximation (measure of inequality)
        # Using top 20 holders as sample
        if len(holders) >= 20:
            shares = sorted([h.get('share', 0) for h in holders[:20]], reverse=True)
            n = len(shares)
            gini = sum([(2 * i - n - 1) * share for i, share in enumerate(shares)])
            gini /= (n * sum(shares)) if sum(shares) > 0 else 1
            metrics['gini_coefficient'] = max(0, min(1, gini))
        else:
            metrics['gini_coefficient'] = 0

        # Concentration risk score (0-100)
        # Higher = more concentrated (riskier)
        concentration_score = (
            min(metrics.get('top10_percentage', 0) / 80 * 40, 40) +  # Top 10 (max 40 pts)
            min(metrics.get('gini_coefficient', 0) * 30, 30) +       # Gini (max 30 pts)
            min(metrics.get('largest_cluster_percentage', 0) / 50 * 20, 20) +  # Clusters (max 20 pts)
            min(metrics.get('contract_percentage', 0) / 20 * 10, 10)  # Contracts (max 10 pts)
        )
        metrics['concentration_risk'] = round(concentration_score, 2)

        # Dev/Insider detection
        # Look for labels like "Dev", "Team", "Insider"
        suspicious_labels = ['dev', 'team', 'insider', 'owner', 'deployer', 'creator']
        dev_holders = []
        for holder in holders:
            label = (holder.get('label') or '').lower()
            if any(sus in label for sus in suspicious_labels):
                dev_holders.append(holder)

        if dev_holders:
            dev_share = sum([h.get('share', 0) for h in dev_holders])
            metrics['dev_percentage'] = dev_share * 100
            metrics['dev_wallet_count'] = len(dev_holders)
        else:
            metrics['dev_percentage'] = 0
            metrics['dev_wallet_count'] = 0

        return metrics

    async def get_multiple_tokens(
        self,
        token_addresses: List[str],
        chain: str = "solana"
    ) -> Dict[str, Optional[Dict]]:
        """
        Obtiene datos de Bubblemaps para múltiples tokens

        Args:
            token_addresses: Lista de contract addresses
            chain: Blockchain

        Returns:
            Dict mapeando address -> map_data
        """
        results = {}

        for token_addr in token_addresses:
            data = await self.get_map_data(token_addr, chain)
            results[token_addr] = data

            # Rate limiting: 1 second between requests
            await asyncio.sleep(1)

        return results


async def test_bubblemaps():
    """Test the Bubblemaps API integration"""
    async with BubblemapsAPI() as api:
        print("=" * 70)
        print("BUBBLEMAPS API TEST")
        print("=" * 70)
        print("\n[*] Testing Bubblemaps API integration...")

        # Test with a known Solana token
        test_token = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"

        print(f"\n[*] Fetching Bubblemaps data for: {test_token[:8]}...")

        data = await api.get_map_data(test_token)

        if data:
            print("\n[✓] Successfully fetched Bubblemaps data:")
            print(f"\nMetadata:")
            print(f"  Last updated: {data.get('metadata', {}).get('last_updated', 'N/A')}")

            print(f"\nHolders:")
            print(f"  Total holders: {data.get('metrics', {}).get('holder_count', 'N/A')}")
            print(f"  Top 1: {data.get('metrics', {}).get('top1_percentage', 0):.2f}%")
            print(f"  Top 10: {data.get('metrics', {}).get('top10_percentage', 0):.2f}%")

            print(f"\nConcentration:")
            print(f"  Gini coefficient: {data.get('metrics', {}).get('gini_coefficient', 0):.3f}")
            print(f"  Risk score: {data.get('metrics', {}).get('concentration_risk', 0)}/100")

            print(f"\nClusters:")
            print(f"  Number of clusters: {data.get('metrics', {}).get('cluster_count', 0)}")
            print(f"  Largest cluster: {data.get('metrics', {}).get('largest_cluster_percentage', 0):.2f}%")

            print(f"\nDev/Team:")
            print(f"  Dev holdings: {data.get('metrics', {}).get('dev_percentage', 0):.2f}%")
            print(f"  Dev wallets: {data.get('metrics', {}).get('dev_wallet_count', 0)}")

            print(f"\nCEX/Contracts:")
            print(f"  CEX: {data.get('metrics', {}).get('cex_percentage', 0):.2f}%")
            print(f"  Contracts: {data.get('metrics', {}).get('contract_percentage', 0):.2f}%")

            # Show top 5 holders
            if data.get('holders'):
                print(f"\nTop 5 Holders:")
                for i, holder in enumerate(data['holders'][:5], 1):
                    label = holder.get('label') or holder.get('address')[:8]
                    share_pct = holder.get('share', 0) * 100
                    is_cex = " [CEX]" if holder.get('is_cex') else ""
                    is_dex = " [DEX]" if holder.get('is_dex') else ""
                    is_contract = " [Contract]" if holder.get('is_contract') else ""
                    print(f"  {i}. {label}: {share_pct:.2f}%{is_cex}{is_dex}{is_contract}")

        else:
            print("\n[!] Failed to fetch Bubblemaps data")


if __name__ == "__main__":
    # Fix Windows encoding
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    asyncio.run(test_bubblemaps())
