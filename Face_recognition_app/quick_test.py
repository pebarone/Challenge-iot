import requests
import json

# Test authentication
print("Testing API authentication...")
response = requests.post(
    'https://assessor-virtual-api-684499909473.southamerica-east1.run.app/api/auth/login',
    json={'email': 'admin@admin.com', 'senha': 'admin'}
)

print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    token = data.get('accessToken')
    print(f"✓ Authentication successful!")
    print(f"Token received: {token[:50]}...")
    
    # Test fetching clients
    print("\nFetching clients...")
    clientes_response = requests.get(
        'https://assessor-virtual-api-684499909473.southamerica-east1.run.app/api/clientes',
        headers={'Authorization': f'Bearer {token}'}
    )
    
    print(f"Status Code: {clientes_response.status_code}")
    
    if clientes_response.status_code == 200:
        clientes = clientes_response.json()
        print(f"✓ Found {len(clientes)} clients")
        
        # Save to file
        output_path = 'c:\\Users\\labsfiap\\Desktop\\Challenge-iot\\clientes.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clientes, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved to: {output_path}")
        print(f"\nFirst client preview:")
        if clientes:
            print(json.dumps(clientes[0], indent=2, ensure_ascii=False))
    else:
        print(f"✗ Error: {clientes_response.text}")
else:
    print(f"✗ Authentication failed: {response.text}")
