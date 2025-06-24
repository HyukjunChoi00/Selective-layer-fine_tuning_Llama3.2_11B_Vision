import pandas as pd
df = pd.read_csv('smiles_with_images_iupac.csv')

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import json
from typing import List, Dict
import os
os.environ['OPENAI_API_KEY'] =  "YOUR API KEY"

def generate_description(row: pd.Series) -> Dict:
    """
    Generate a detailed explanation using the information from a single row of the DataFrame.
    """
    template = """
You are an expert in organic chemistry nomenclature, creating training data for a Vision Language Model (VLM) that will learn to name chemical compounds from their structural images.

For the compound with:
IUPAC Name: {iupac_name}
SMILES: {smiles}

Create a detailed explanation that teaches the VLM how to identify and name this molecule from its **visual structure only**. Your explanation should follow this pattern:

---

1. Start with "Looking at this molecule's structure, I can see..." and complete that sentence naturally.
- For example: "Looking at this molecule's structure, I can see a six-membered aromatic ring with alternating double bonds, indicating a benzene ring."
Describe the visible structural features in terms of what can be **seen** in the image:
   - Overall shape: Is the molecule linear, branched, cyclic, or polycyclic?
   - Ring systems: How many rings are visible? Are any rings fused together? 
   - Vertex counting: How many vertices (carbon atoms) are present in each ring or chain?  
   - Atom types: Are there any labeled atoms such as N, O, S, Cl, etc.?  
   - Functional groups: Can you visually identify groups like -OH, -NO₂, -COOH, -SO₂NH₂, etc.?  
   - Symmetry or spatial layout: Is there a plane of symmetry or repeating pattern?

---

2. **Explain the naming process based on these visual cues**:
   - Identify the main ring or chain based on size and connectivity  
   - Explain how to number the atoms: e.g., "starting at the nitrogen and moving clockwise..."  
   - Point out the position of each substituent using visual features  
   - Discuss how multiple rings or fused systems are interpreted  
   - Connect substituents to their naming: e.g., "the -OCH₂CH₃ group is called 'ethoxy' and is located at position 6"

---

3. **End with a clear conclusion**:
   Summarize how the observed structure leads to the final name, in this format:  
   > "Based on these structural features, this molecule is named {iupac_name}. Its SMILES representation is {smiles}."

---

**Key instruction**:
Focus on **visual teaching** — you are guiding a model that only sees the image. Avoid chemical theory unless it can be inferred visually. Describe what a model with vision can observe: **shapes, positions, labels, bond patterns, ring fusion, symmetry**.

Example style:
> "Looking at this molecule's structure, I can see a six-membered hexagonal ring with alternating single and double bonds, forming a benzene ring. There is a -OH group attached to the top vertex, and a -NO₂ group at the bottom left vertex..."

"""
    
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")
    
    try:
        # 각 행의 값을 직접 접근
        smiles_val = row['smiles']
        iupac_val = row['IUPAC_Name']
        
        messages = prompt.format_messages(
            iupac_name=iupac_val,
            smiles=smiles_val
        )
        
        response = model.invoke(messages)
        return {
            'smiles': smiles_val,
            'iupac_name': iupac_val,
            'description': response.content
        }
        
    except Exception as e:
        print(f"Error processing row: {e}")
        return {
            'smiles': row['smiles'] if 'smiles' in row else None,
            'iupac_name': row['IUPAC_Name'] if 'IUPAC_Name' in row else None,
            'error': str(e)
        }

def process_dataframe(df: pd.DataFrame) -> List[Dict]:
    """
    데이터프레임을 처리하여 설명을 생성합니다.
    """
    results = []
    
    # 인덱스 재설정
    df_reset = df.reset_index(drop=True)
    
    for i, row in df_reset.iterrows():
        print(f"처리 중 {i+1}/{len(df_reset)}: {row['smiles']}")
        result = generate_description(row)
        results.append(result)
    
    return results

if __name__ == "__main__":
    # 테스트용으로 처음 2개 행만 선택
    test_df = df.head(2).copy()  # 안전한 복사본 생성
    
    print("테스트 데이터 확인:")
    print(test_df)
    print("\n열 이름:", test_df.columns.tolist())
    
    print("\n테스트 실행 (2개 항목):")
    print("-" * 50)
    
    results = process_dataframe(test_df)
    
    # 결과 출력
    for result in results:
        print("\nSMILES:", result['smiles'])
        print("IUPAC:", result['iupac_name'])
        if 'description' in result:
            print("\n설명:")
            print(result['description'])
        else:
            print("오류:", result.get('error', 'Unknown error'))
        print("-" * 50)
    
    proceed = input("\n테스트 결과가 만족스러운가요? 전체 데이터셋을 처리하시겠습니까? (y/n): ")
    
    if proceed.lower() == 'y':
        print(f"\n전체 데이터셋 처리 시작 ({len(df)}개 항목)")
        all_results = process_dataframe(df)
        
        # 결과 저장
        with open("chemical_descriptions.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n완료: 결과가 'chemical_descriptions.json'에 저장되었습니다.")
    else:
        print("\n처리가 취소되었습니다.")
