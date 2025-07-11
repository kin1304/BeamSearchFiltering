"""
Improved Scoring Methods for Beam Search với Contradiction Detection
Các phương pháp cải thiện để tăng độ liên quan và phát hiện mâu thuẫn giữa sentences và claim
"""

from typing import List, Set, Dict, Any, Optional
from difflib import SequenceMatcher
import re
import math


class ImprovedScoring:
    """Class chứa các methods cải thiện scoring cho beam search với contradiction detection"""
    
    def __init__(self):
        # Enhanced scoring weights
        self.word_match_weight = 5.0        # Tăng trọng số cho exact match
        self.semantic_match_weight = 3.0    # Semantic similarity
        self.fuzzy_match_weight = 2.0       # Fuzzy string matching
        self.entity_bonus = 2.5             # Entity relevance
        self.length_penalty = 0.05          # Giảm penalty cho path dài
        self.sentence_bonus = 1.5           # Bonus cho mỗi sentence
        
        # ✅ MỚI: Contradiction & Negation weights
        self.contradiction_weight = 4.0     # Trọng số cho contradiction
        self.negation_weight = 3.5         # Trọng số cho negation
        self.opposite_weight = 2.5         # Trọng số cho opposite meaning
        self.temporal_contradiction_weight = 3.0  # Thời gian mâu thuẫn
        
        # Vietnamese negation indicators
        self.negation_words = {
            'không', 'chẳng', 'chả', 'chưa', 'chớ', 'đừng', 'không phải',
            'chưa từng', 'chẳng bao giờ', 'không bao giờ', 'không hề',
            'không thể', 'không nên', 'không được', 'chưa bao giờ',
            'không còn', 'không có', 'thiếu', 'mất', 'hết', 'vắng',
            'trống', 'rỗng', 'không tồn tại', 'không xuất hiện'
        }
        
        # Contradiction indicators
        self.contradiction_words = {
            'nhưng', 'tuy nhiên', 'mặc dù', 'dù', 'dù cho', 'cho dù',
            'trái lại', 'ngược lại', 'thực tế', 'thật ra', 'sự thật là',
            'thay vào đó', 'trong khi', 'về thực tế', 'thay vì',
            'tương phản', 'khác với', 'khác biệt', 'mâu thuẫn',
            'bác bỏ', 'phản bác', 'cãi lại', 'đối lập'
        }
        
        # Opposite meaning pairs (claim -> contradiction)
        self.opposite_pairs = {
            'lớn': ['nhỏ', 'bé', 'tí', 'ít', 'tí hon', 'nhỏ xíu'],
            'nhiều': ['ít', 'thiếu', 'không có', 'hết', 'vài', 'mấy'],
            'có': ['không có', 'thiếu', 'mất', 'hết', 'vắng', 'trống'],
            'được': ['không được', 'bị cấm', 'không thể', 'bị từ chối'],
            'thành công': ['thất bại', 'thảm bại', 'không thành công', 'thua', 'hỏng'],
            'tăng': ['giảm', 'sụt', 'rơi', 'xuống', 'giảm sút', 'suy giảm'],
            'tốt': ['xấu', 'tệ', 'kém', 'dở', 'tồi tệ', 'thảm hại'],
            'mới': ['cũ', 'cũ kỹ', 'lỗi thời', 'lạc hậu', 'cổ'],
            'nhanh': ['chậm', 'lâu', 'trễ', 'từ từ', 'chậm chạp'],
            'đúng': ['sai', 'nhầm', 'lỗi', 'false', 'không chính xác'],
            'cao': ['thấp', 'alp', 'thấp lè', 'gần đất'],
            'dài': ['ngắn', 'cụt', 'tít', 'ngắn cũn'],
            'rộng': ['hẹp', 'chật', 'nhỏ hẹp', 'chật chội'],
            'nóng': ['lạnh', 'mát', 'lạnh lẽo', 'băng giá'],
            'sáng': ['tối', 'tối tăm', 'u ám', 'mờ'],
            'mạnh': ['yếu', 'ốm', 'yếu ớt', 'bé nhỏ']
        }
        
        # Advanced scoring weights
        self.synonym_weight = 2.0           # Từ đồng nghĩa
        self.tfidf_weight = 1.5            # TF-IDF similarity
        self.position_weight = 1.0         # Vị trí của từ trong câu
        
    def enhanced_word_matching(self, claim_words: Set[str], path_words: Set[str]) -> float:
        """
        Cải thiện word matching với nhiều phương pháp
        """
        if not claim_words or not path_words:
            return 0.0
            
        # 1. Exact match (như trước)
        exact_matches = claim_words.intersection(path_words)
        exact_ratio = len(exact_matches) / len(claim_words)
        
        # 2. Partial matching (từ con)
        partial_matches = 0
        for claim_word in claim_words:
            for path_word in path_words:
                if len(claim_word) >= 3 and len(path_word) >= 3:
                    if claim_word in path_word or path_word in claim_word:
                        partial_matches += 1
                        break
        partial_ratio = partial_matches / len(claim_words)
        
        # 3. Edit distance matching
        edit_distance_matches = 0
        for claim_word in claim_words:
            for path_word in path_words:
                if len(claim_word) >= 3 and len(path_word) >= 3:
                    similarity = SequenceMatcher(None, claim_word, path_word).ratio()
                    if similarity >= 0.8:  # 80% tương tự
                        edit_distance_matches += 1
                        break
        edit_ratio = edit_distance_matches / len(claim_words)
        
        # Combine scores với trọng số
        total_score = (exact_ratio * 1.0 + 
                      partial_ratio * 0.6 + 
                      edit_ratio * 0.4)
        
        return total_score
        
    def semantic_similarity_score(self, claim_words: Set[str], path_words: Set[str]) -> float:
        """
        Tính semantic similarity với nhiều metrics
        """
        if not claim_words or not path_words:
            return 0.0
            
        # 1. Jaccard similarity
        intersection = claim_words.intersection(path_words)
        union = claim_words.union(path_words)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # 2. Cosine similarity (simplified)
        dot_product = len(intersection)
        magnitude_claim = math.sqrt(len(claim_words))
        magnitude_path = math.sqrt(len(path_words))
        cosine = dot_product / (magnitude_claim * magnitude_path) if (magnitude_claim * magnitude_path) > 0 else 0.0
        
        # 3. Overlap coefficient
        overlap = len(intersection) / min(len(claim_words), len(path_words)) if min(len(claim_words), len(path_words)) > 0 else 0.0
        
        # Combine metrics
        semantic_score = (jaccard * 0.3 + cosine * 0.4 + overlap * 0.3)
        return semantic_score
        
    def fuzzy_text_similarity(self, claim_text: str, sentence_text: str) -> float:
        """
        Enhanced fuzzy matching cho toàn bộ text
        """
        if not claim_text or not sentence_text:
            return 0.0
            
        # Normalize texts
        claim_norm = self._normalize_text(claim_text)
        sentence_norm = self._normalize_text(sentence_text)
        
        # 1. Overall similarity
        overall_sim = SequenceMatcher(None, claim_norm, sentence_norm).ratio()
        
        # 2. Longest common subsequence ratio
        lcs_ratio = self._lcs_ratio(claim_norm, sentence_norm)
        
        # 3. N-gram similarity (bigrams)
        bigram_sim = self._ngram_similarity(claim_norm, sentence_norm, n=2)
        
        # Combine scores
        fuzzy_score = (overall_sim * 0.5 + lcs_ratio * 0.3 + bigram_sim * 0.2)
        return fuzzy_score

    def detect_negation_patterns(self, claim_text: str, sentence_text: str) -> float:
        """
        ✅ MỚI: Phát hiện patterns phủ định giữa claim và sentence
        """
        if not claim_text or not sentence_text:
            return 0.0
            
        claim_words = claim_text.lower().split()
        sentence_words = sentence_text.lower().split()
        
        negation_score = 0.0
        
        # 1. Direct negation: Claim có từ khẳng định, sentence có từ phủ định
        claim_has_negation = any(word in self.negation_words for word in claim_words)
        sentence_has_negation = any(word in self.negation_words for word in sentence_words)
        
        if not claim_has_negation and sentence_has_negation:
            # Claim khẳng định, sentence phủ định -> potential contradiction
            shared_words = set(claim_words).intersection(set(sentence_words))
            if len(shared_words) >= 2:  # Có ít nhất 2 từ chung
                negation_score += 0.8
                
        elif claim_has_negation and not sentence_has_negation:
            # Claim phủ định, sentence khẳng định -> potential contradiction
            shared_words = set(claim_words).intersection(set(sentence_words))
            if len(shared_words) >= 2:
                negation_score += 0.8
        
        # 2. Opposite word detection
        for claim_word in claim_words:
            if claim_word in self.opposite_pairs:
                opposite_words = self.opposite_pairs[claim_word]
                if any(opp_word in sentence_words for opp_word in opposite_words):
                    negation_score += 0.6
                    
        # 3. Numbers contradiction (quan trọng cho fact-checking)
        claim_numbers = self._extract_numbers(claim_text)
        sentence_numbers = self._extract_numbers(sentence_text)
        
        if claim_numbers and sentence_numbers:
            # Kiểm tra số khác nhau đáng kể
            for c_num in claim_numbers:
                for s_num in sentence_numbers:
                    # ✅ BẢO VỆ KHỎI DIVISION BY ZERO
                    max_num = max(c_num, s_num)
                    if max_num > 0 and abs(c_num - s_num) / max_num > 0.3:  # Khác nhau >30%
                        negation_score += 0.4
        
        # 4. Temporal contradiction (thời gian mâu thuẫn)
        temporal_score = self._detect_temporal_contradiction(claim_text, sentence_text)
        negation_score += temporal_score * 0.3
        
        return min(negation_score, 1.0)
        
    def detect_contradiction_patterns(self, claim_text: str, sentence_text: str) -> float:
        """
        ✅ MỚI: Phát hiện patterns mâu thuẫn
        """
        if not claim_text or not sentence_text:
            return 0.0
            
        sentence_words = sentence_text.lower().split()
        contradiction_score = 0.0
        
        # 1. Contradiction indicators trong sentence
        has_contradiction_words = any(word in self.contradiction_words for word in sentence_words)
        if has_contradiction_words:
            # Kiểm tra xem có overlap với claim không
            claim_words = set(claim_text.lower().split())
            sentence_words_set = set(sentence_words)
            shared_words = claim_words.intersection(sentence_words_set)
            
            if len(shared_words) >= 2:
                contradiction_score += 0.7
                
        # 2. Pattern "X nhưng Y" hoặc "Mặc dù X, Y"
        sentence_lower = sentence_text.lower()
        contradiction_patterns = [
            r'(.+)\s+(nhưng|tuy nhiên|mặc dù)\s+(.+)',
            r'(mặc dù|dù|dù cho)\s+(.+),\s*(.+)',
            r'(.+)\s+(trái lại|ngược lại)\s+(.+)',
            r'(thực tế|thật ra|sự thật là)\s+(.+)',
            r'(.+)\s+(khác với|khác biệt|mâu thuẫn)\s+(.+)'
        ]
        
        for pattern in contradiction_patterns:
            match = re.search(pattern, sentence_lower)
            if match:
                # Kiểm tra parts có relate đến claim không
                parts = [part.strip() for part in match.groups() if part]
                claim_words = set(claim_text.lower().split())
                
                for part in parts:
                    part_words = set(part.split())
                    if len(claim_words.intersection(part_words)) >= 1:
                        contradiction_score += 0.5
                        break
                        
        # 3. Percentage/ratio contradiction
        claim_percentages = self._extract_percentages(claim_text)
        sentence_percentages = self._extract_percentages(sentence_text)
        
        if claim_percentages and sentence_percentages:
            for c_pct in claim_percentages:
                for s_pct in sentence_percentages:
                    if abs(c_pct - s_pct) > 20:  # Khác nhau >20%
                        contradiction_score += 0.4
                        
        return min(contradiction_score, 1.0)
        
    def _detect_temporal_contradiction(self, claim_text: str, sentence_text: str) -> float:
        """Phát hiện mâu thuẫn về thời gian"""
        temporal_words = {
            'trước': ['sau', 'kế tiếp', 'tiếp theo'],
            'sau': ['trước', 'trước đó'],
            'sớm': ['muộn', 'trễ', 'chậm'],
            'muộn': ['sớm', 'nhanh', 'kịp thời'],
            'cũ': ['mới', 'hiện đại', 'tương lai'],
            'mới': ['cũ', 'xưa', 'cổ điển']
        }
        
        claim_words = claim_text.lower().split()
        sentence_words = sentence_text.lower().split()
        
        temporal_score = 0.0
        for claim_word in claim_words:
            if claim_word in temporal_words:
                opposite_temps = temporal_words[claim_word]
                if any(temp in sentence_words for temp in opposite_temps):
                    temporal_score += 0.5
                    
        return min(temporal_score, 1.0)
        
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers từ text"""
        numbers = []
        
        # Extract integer và float numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        matches = re.findall(number_pattern, text)
        
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
                
        return numbers
        
    def _extract_percentages(self, text: str) -> List[float]:
        """Extract percentages từ text"""
        percentages = []
        
        # Pattern cho phần trăm
        percentage_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*phần\s*trăm',
            r'(\d+(?:\.\d+)?)\s*per\s*cent'
        ]
        
        for pattern in percentage_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    percentages.append(float(match))
                except ValueError:
                    continue
                    
        return percentages
        
    def entity_relevance_score(self, claim_entities: Set[str], path_entities: Set[str]) -> float:
        """
        Tính độ liên quan của entities
        """
        if not claim_entities:
            return 0.0
            
        if not path_entities:
            return 0.0
            
        # Entity overlap
        entity_matches = claim_entities.intersection(path_entities)
        entity_ratio = len(entity_matches) / len(claim_entities)
        
        # Bonus cho số lượng entities matched
        entity_count_bonus = min(len(entity_matches) * 0.2, 1.0)
        
        return entity_ratio + entity_count_bonus
        
    def sentence_quality_score(self, sentence_text: str, claim_text: str) -> float:
        """
        Đánh giá chất lượng và độ liên quan của sentence
        """
        if not sentence_text or not claim_text:
            return 0.0
            
        # 1. Length appropriateness (không quá ngắn hoặc quá dài)
        sentence_len = len(sentence_text.split())
        claim_len = len(claim_text.split())
        
        if sentence_len < 3:  # Quá ngắn
            length_score = 0.2
        elif sentence_len > claim_len * 3:  # Quá dài
            length_score = 0.6
        else:
            length_score = 1.0
            
        # 2. Information density (số từ có nghĩa)
        meaningful_words = [word for word in sentence_text.lower().split() 
                          if len(word) > 2 and word.isalpha()]
        density_score = min(len(meaningful_words) / max(sentence_len, 1), 1.0)
        
        # 3. Sentence structure score (có chứa các từ kết nối, động từ, etc.)
        structure_score = self._sentence_structure_score(sentence_text)
        
        # Combine scores
        quality_score = (length_score * 0.4 + density_score * 0.3 + structure_score * 0.3)
        return quality_score
        
    def calculate_enhanced_score_with_contradiction(self, 
                                                  claim_text: str, 
                                                  claim_words: Set[str],
                                                  path_words: Set[str],
                                                  sentence_texts: List[str],
                                                  entity_count: int,
                                                  path_length: int,
                                                  claim_entities: Optional[Set[str]] = None) -> float:
        """
        ✅ ENHANCED: Tính score bao gồm cả contradiction detection
        """
        # Base score từ method cũ
        base_score = self.calculate_enhanced_score(
            claim_text, claim_words, path_words, sentence_texts, entity_count, path_length, claim_entities
        )
        
        # ✅ MỚI: Contradiction và Negation scoring
        contradiction_bonus = 0.0
        
        for sentence_text in sentence_texts:
            # Negation detection
            negation_score = self.detect_negation_patterns(claim_text, sentence_text)
            contradiction_bonus += negation_score * self.negation_weight
            
            # Contradiction detection  
            contradiction_score = self.detect_contradiction_patterns(claim_text, sentence_text)
            contradiction_bonus += contradiction_score * self.contradiction_weight
        
        # Combine scores
        total_score = base_score + contradiction_bonus
        
        return total_score
        
    def calculate_enhanced_score(self, 
                                claim_text: str, 
                                claim_words: Set[str],
                                path_words: Set[str],
                                sentence_texts: List[str],
                                entity_count: int,
                                path_length: int,
                                                                 claim_entities: Optional[Set[str]] = None) -> float:
        """
        Tính tổng điểm được cải thiện cho một path
        """
        total_score = 0.0
        
        # 1. Enhanced word matching
        word_score = self.enhanced_word_matching(claim_words, path_words)
        total_score += word_score * self.word_match_weight
        
        # 2. Semantic similarity
        semantic_score = self.semantic_similarity_score(claim_words, path_words)
        total_score += semantic_score * self.semantic_match_weight
        
        # 3. Fuzzy matching với sentences
        if sentence_texts:
            max_fuzzy_score = 0.0
            claim_entity_boost = 0.0
            
            for sentence_text in sentence_texts:
                fuzzy_score = self.fuzzy_text_similarity(claim_text, sentence_text)
                quality_score = self.sentence_quality_score(sentence_text, claim_text)
                
                # ✅ MỚI: Boost cho sentences chứa claim entities
                entity_boost = 0.0
                if claim_entities:
                    sentence_lower = sentence_text.lower()
                    claim_entity_matches = sum(1 for entity in claim_entities if entity.lower() in sentence_lower)
                    entity_boost = (claim_entity_matches / len(claim_entities)) * 0.5  # Boost tối đa 0.5
                
                combined_sentence_score = (fuzzy_score + quality_score + entity_boost) / 2
                max_fuzzy_score = max(max_fuzzy_score, combined_sentence_score)
                claim_entity_boost = max(claim_entity_boost, entity_boost)
                
            total_score += max_fuzzy_score * self.fuzzy_match_weight
            # Thêm bonus riêng cho claim entities
            total_score += claim_entity_boost * 2.0
            
        # 4. Entity bonus
        total_score += entity_count * self.entity_bonus
        
        # 5. Length penalty (giảm)
        total_score -= path_length * self.length_penalty
        
        # 6. Sentence count bonus
        sentence_count = len(sentence_texts)
        total_score += sentence_count * self.sentence_bonus
        
        return total_score
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text để so sánh"""
        # Lowercase và remove extra spaces
        text = text.lower().strip()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text
        
    def _lcs_ratio(self, text1: str, text2: str) -> float:
        """Tính Longest Common Subsequence ratio"""
        def lcs_length(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        lcs_len = lcs_length(text1, text2)
        max_len = max(len(text1), len(text2))
        return lcs_len / max_len if max_len > 0 else 0.0
        
    def _ngram_similarity(self, text1: str, text2: str, n: int = 2) -> float:
        """Tính N-gram similarity"""
        def get_ngrams(text, n):
            words = text.split()
            return set([' '.join(words[i:i+n]) for i in range(len(words)-n+1)])
        
        if len(text1.split()) < n or len(text2.split()) < n:
            return 0.0
            
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
            
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        
        return len(intersection) / len(union) if union else 0.0
        
    def _sentence_structure_score(self, sentence: str) -> float:
        """Đánh giá cấu trúc câu"""
        # Các từ kết nối và quan trọng
        connectors = {'và', 'nhưng', 'tuy nhiên', 'do đó', 'vì vậy', 'bởi vì', 'nếu', 'khi', 'mà', 'để'}
        verbs_indicators = {'là', 'có', 'được', 'sẽ', 'đã', 'đang', 'bị', 'cho', 'từ', 'trong'}
        
        words = sentence.lower().split()
        
        # Check for connectors
        has_connectors = any(word in connectors for word in words)
        
        # Check for verb indicators
        has_verbs = any(word in verbs_indicators for word in words)
        
        # Check for balanced structure
        has_balanced_length = 5 <= len(words) <= 30
        
        # Combine factors
        structure_score = 0.0
        if has_connectors:
            structure_score += 0.4
        if has_verbs:
            structure_score += 0.4
        if has_balanced_length:
            structure_score += 0.2
            
        return min(structure_score, 1.0) 