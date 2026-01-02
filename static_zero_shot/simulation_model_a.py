import os
import time
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: .env 파일을 찾을 수 없거나 API 키가 없습니다.")
    exit()

client = OpenAI(api_key=api_key)

# ==========================================
# 1. 통계 상수 정의 (Source: ESA 2024, Statista, Newzoo)
# ==========================================
GENDER_KEYS = ["Male", "Female"]
GENDER_PROBS = [0.54, 0.46]

AGE_KEYS = [
    "18-19 (Late Teens)", 
    "20-29 (Young Adults)", 
    "30-39 (Core Millennials)", 
    "40-49 (Gen X)", 
    "50-59 (Older Gen X)", 
    "60-64 (Boomers)"
]
AGE_PROBS = [0.04, 0.24, 0.26, 0.21, 0.17, 0.08]

GAMER_PERSONA_DATA = {
    "The Ultimate Gamer": {
        "prob": 0.13,
        "desc": "You spend all your money and free time on games. You love high-end hardware and buy AAA titles immediately."
    },
    "The All-Round Enthusiast": {
        "prob": 0.09,
        "desc": "You enjoy a balanced gaming life. You carefully read reviews and analyze the game quality before buying."
    },
    "The Cloud Gamer": {
        "prob": 0.19,
        "desc": "You prefer streaming and discounted games. You don't have high-end hardware and value accessibility."
    },
    "The Conventional Player": {
        "prob": 0.04,
        "desc": "You only play familiar franchises (FIFA, CoD). You are indifferent to new IPs."
    },
    "The Hardware Enthusiast": {
        "prob": 0.09,
        "desc": "You follow the latest tech trends. You buy games to test your rig's graphical capabilities."
    },
    "The Popcorn Gamer": {
        "prob": 0.13,
        "desc": "You prefer watching Twitch/YouTube streams over playing. You get satisfaction from watching others play."
    },
    "The Backseat Gamer": {
        "prob": 0.06,
        "desc": "You used to game a lot, but now you only watch content due to lack of time."
    },
    "The Time Filler": {
        "prob": 0.27,
        "desc": "You play mobile games to kill time. You rarely buy expensive console/PC games."
    }
}

PERSONA_KEYS = list(GAMER_PERSONA_DATA.keys())
PERSONA_PROBS = [GAMER_PERSONA_DATA[k]["prob"] for k in PERSONA_KEYS]

# ==========================================
# 2. 에이전트 생성 함수
# ==========================================

def generate_agent(agent_id):
    """
    통계적 확률에 기반하여 가상 에이전트 하나를 생성.
    """
    gender = np.random.choice(GENDER_KEYS, p=GENDER_PROBS)
    age = np.random.choice(AGE_KEYS, p=AGE_PROBS)
    
    # 페르소나 뽑기
    p_type = np.random.choice(PERSONA_KEYS, p=PERSONA_PROBS)
    p_desc = GAMER_PERSONA_DATA[p_type]["desc"]
    
    return {
        "id": agent_id,
        "gender": gender,
        "age": age,
        "type": p_type,
        "desc": p_desc
    }

# ==========================================
# 3. 메인 시뮬레이션 루프
# ==========================================

def run_simulation(total_agents=100):
    print(f"--- Team 1: Static Zero-Shot Simulation ({total_agents} Agents) ---")
    print("Goal: Predict purchase intent for 'Cyberpunk 2077' WITHOUT external context.\n")
    
    results = []

    for i in range(total_agents):
        # Step 3.1: 에이전트 생성
        agent = generate_agent(i + 1)
        
        # Step 3.2: 프롬프트 구성
        system_prompt = f"""
        You are a {agent['age']} {agent['gender']}.
        Your gamer personality is '{agent['type']}'.
        Description: {agent['desc']}
        
        Act strictly according to this persona.
        Rely ONLY on your internal knowledge and personal preferences.
        Do NOT assume any recent news regarding bugs or refunds unless your persona specifically cares about pre-release hype.
        """
        
        user_prompt = "Will you buy the video game 'Cyberpunk 2077'? Answer starting with YES or NO, followed by a short reason."

        try:
            # Step 3.3: LLM 호출
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )
            
            full_response = response.choices[0].message.content
            
            # Step 3.4: 결과 파싱 (YES/NO 추출)
            decision = "UNKNOWN"
            if full_response.strip().upper().startswith("YES"):
                decision = "YES"
            elif full_response.strip().upper().startswith("NO"):
                decision = "NO"
            
            print(f"[{i+1}/{total_agents}] {agent['type']} ({agent['age']}) -> {decision}")
            
            # 결과 리스트에 추가
            results.append({
                "Agent_ID": agent['id'],
                "Gender": agent['gender'],
                "Age": agent['age'],
                "Persona_Type": agent['type'],
                "Decision": decision,
                "Reasoning": full_response,
                "System_Prompt": system_prompt
            })

        except Exception as e:
            print(f"Error on Agent {i+1}: {e}")
            time.sleep(1)

    # Step 3.5: 결과 저장
    df = pd.DataFrame(results)
    filename = "Team1_Static_ZeroShot_Results.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    
    print(f"\nSimulation Complete. Results saved to '{filename}'.")
    
    # 간단한 통계 출력
    print("\n[Summary]")
    print(df['Decision'].value_counts(normalize=True))

# ==========================================
# 4. 실행 부
# ==========================================

if __name__ == "__main__":
    run_simulation(100)

