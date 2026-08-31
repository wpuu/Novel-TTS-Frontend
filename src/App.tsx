import React, { useState, useEffect, useRef, useCallback } from 'react';
import { cn } from './utils/cn';
import { 
  Play, Pause, AlertCircle, CheckCircle2, FileJson, 
  Settings, Users, RefreshCw, Volume2, Download, 
  Upload, Terminal, Activity, Edit3, Music, Loader2, Sparkles,
  Layers, ChevronRight, Check, AlertTriangle, FileText,
  PanelRightClose, PanelRightOpen, FolderOpen, Trash2, Plus
} from 'lucide-react';

// ==========================================
// 1. 核心数据结构与接口 (Types & Interfaces)
// ==========================================

interface CastMember {
  speaker_id?: string;
  character_name: string;
  character_name_en?: string;
  aliases: string[];
  gender?: string;
  core?: boolean;
  shared_voice?: boolean;
  assigned_voice_id: string;
  speaker_alias?: string;
  voice_profile?: string;
  voice_reason?: string;
  tags_suggestion?: string[];
  style_instructions?: string;
  colorHex?: string; // Generated for UI
}

interface CastJson {
  novel_name: string;
  source_author?: string;
  tts_model?: string;
  tagging_model?: string;
  voice_conflict_check?: string;
  pov_instruction?: string;
  cast: CastMember[];
}

interface TaggedSegment {
  seq?: number;
  chapter?: string;
  type?: string;
  char: string;
  speaker_id?: string;
  speaker_alias?: string;
  voice: string;
  char_in_cast?: boolean;
  char_inferred?: boolean;
  emotion_class: string;
  audio_tag: string;
  text: string;
}

interface ScriptChunk {
  id: string;
  index: number;
  originalText: string;
  previousTextContext: string;
  status: 'pending' | 'processing' | 'success' | 'failed' | 'rescued';
  segments: TaggedSegment[];
  error?: string;
}

interface KeyRuntimeState {
  key: string;
  isRateLimited: boolean;
  cooldownUntil: number;
}

interface TTSBatch {
  id: string;
  speaker: string;
  voiceId: string;
  emotionClass: string;
  text: string; // Concatenated text with pause anchors
  seqs: number[]; // global seq IDs included
  status: 'pending' | 'generating' | 'success' | 'failed';
  error?: string;
  audioBlob?: Blob;
  audioUrl?: string;
  slicedAudios?: { seq: number; blob: Blob; url: string; fallbackLevel?: number }[];
  ttsPrompt?: string;
  totalBytes?: number;
  anchorMap?: any[];
}

interface ProjectMeta {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  totalChunks: number;
}

// ==========================================
// 2. 辅助算法：字符编码检测与智能切片
// ==========================================

// 自动检测 UTF-8 与 GBK
const isUtf8 = (buf: Uint8Array): boolean => {
  let i = 0;
  while (i < buf.length) {
    if (buf[i] <= 0x7F) { i += 1; continue; }
    if (buf[i] >= 0xC2 && buf[i] <= 0xDF) {
      if (i + 1 < buf.length && buf[i+1] >= 0x80 && buf[i+1] <= 0xBF) { i += 2; continue; }
      return false;
    }
    if (buf[i] >= 0xE0 && buf[i] <= 0xEF) {
      if (i + 2 < buf.length && 
          buf[i+1] >= 0x80 && buf[i+1] <= 0xBF && 
          buf[i+2] >= 0x80 && buf[i+2] <= 0xBF) { i += 3; continue; }
      return false;
    }
    if (buf[i] >= 0xF0 && buf[i] <= 0xF4) {
      if (i + 3 < buf.length && 
          buf[i+1] >= 0x80 && buf[i+1] <= 0xBF && 
          buf[i+2] >= 0x80 && buf[i+2] <= 0xBF && 
          buf[i+3] >= 0x80 && buf[i+3] <= 0xBF) { i += 4; continue; }
      return false;
    }
    return false;
  }
  return true;
};

// 智能切片算法：按句就近切割，带 contextBefore 保证语义
const smartChunking = (text: string, targetSize: number = 1000, _margin: number = 200): { originalText: string, previousTextContext: string }[] => {
  const chunks: { originalText: string, previousTextContext: string }[] = [];
  if (!text) return chunks;

  // 1. 防御非法输入：0/负数/NaN 会让 avgSize 归零、切割游标停滞导致死循环冻结页面
  const safeSize = Math.max(100, Math.floor(targetSize) || 100);
  const totalChunks = Math.max(1, Math.ceil(text.length / safeSize));
  const avgSize = Math.max(1, Math.floor(text.length / totalChunks));

  // 2. 预处理安全点：绝对不能在引号内部进行切割
  const isSafe = new Uint8Array(text.length);
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '“') inQuotes = true;
    else if (char === '”') inQuotes = false;
    else if (char === '"') inQuotes = !inQuotes; // 英文引号翻转
    
    // 如果当前不在引号内，则标记为安全 (1)
    isSafe[i] = inQuotes ? 0 : 1;
  }

  let startIdx = 0;
  
  while (startIdx < text.length) {
    // 剩余字数如果只比平均值多一点（比如 1.3 倍），直接全部吃下，防止最后剩一个小尾巴
    if (text.length - startIdx <= avgSize * 1.3) {
      chunks.push({
        originalText: text.substring(startIdx),
        previousTextContext: startIdx === 0 ? '' : text.substring(Math.max(0, startIdx - 200), startIdx)
      });
      break;
    }

    const targetIdx = startIdx + avgSize;
    // 允许在 avgSize 的 0.5倍 到 1.5倍 之间寻找最佳切割点
    const minSearch = startIdx + Math.floor(avgSize * 0.5);
    const maxSearch = Math.min(startIdx + Math.floor(avgSize * 1.5), text.length);
    
    let bestIdx = -1;
    let bestScore = -Infinity;

    // 寻找最佳断点：优先整段（换行符），其次整句（标点符）
    for (let i = minSearch; i <= maxSearch; i++) {
      if (i === 0) continue;
      // 必须确保上一个字符的标记是“安全的”（不在双引号内部）
      if (isSafe[i - 1] === 0) continue;

      const prevChar = text[i - 1];
      const prevPrevChar = i >= 2 ? text[i - 2] : '';
      
      const isParaBreak = prevChar === '\n';
      // 句号、感叹号、问号、省略号。如果紧跟着右双引号，则断点在双引号之后
      const isSentBreak = ['。', '！', '？', '…', '.', '!', '?'].includes(prevChar) || 
                          ((prevChar === '”' || prevChar === '"') && ['。', '！', '？', '…', '.', '!', '?'].includes(prevPrevChar));

      if (isParaBreak || isSentBreak) {
        // 计算距离目标字数的偏差（越小越好）
        const dist = Math.abs(i - targetIdx);
        // 整段优先：给予换行符极高的加分，强制优先在换行处切断
        const typeBonus = isParaBreak ? 10000 : 0;
        const score = typeBonus - dist;
        
        if (score > bestScore) {
          bestScore = score;
          bestIdx = i;
        }
      }
    }

    // 如果在这个区间内居然连一个标点都没找到（极小概率），只能硬切
    if (bestIdx === -1) {
      bestIdx = targetIdx;
    }

    chunks.push({
      originalText: text.substring(startIdx, bestIdx),
      previousTextContext: startIdx === 0 ? '' : text.substring(Math.max(0, startIdx - 200), startIdx)
    });
    
    startIdx = bestIdx;
  }

  return chunks;
};

// ==========================================
// 3. 音频核心逻辑：WAV头封装、JS静音检测、Proportional切分
// ==========================================

// 将 PCM bytes (24kHz Mono 16-bit) 转换为带有 44 字节头的完整 WAV Blob
const pcmToWav = (pcmBytes: Uint8Array, sampleRate: number = 24000): Blob => {
  const buffer = new ArrayBuffer(44 + pcmBytes.length);
  const view = new DataView(buffer);
  
  view.setUint32(0, 0x52494646, false); // "RIFF"
  view.setUint32(4, 36 + pcmBytes.length, true); // total size
  view.setUint32(8, 0x57415645, false); // "WAVE"
  view.setUint32(12, 0x666d7420, false); // "fmt "
  view.setUint32(16, 16, true); // chunk size
  view.setUint16(20, 1, true); // format = PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (sampleRate * bitsPerSample / 8)
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  view.setUint32(36, 0x64617461, false); // "data"
  view.setUint32(40, pcmBytes.length, true);
  
  const wavBytes = new Uint8Array(buffer);
  wavBytes.set(pcmBytes, 44);
  
  return new Blob([wavBytes], { type: 'audio/wav' });
};

// 提取 AudioBuffer 的某个时间区间并封装为 WAV Blob
const bufferToWav = (audioBuffer: AudioBuffer, startSec: number, endSec: number): Blob => {
  const sampleRate = audioBuffer.sampleRate;
  const startSample = Math.floor(startSec * sampleRate);
  const endSample = Math.min(Math.floor(endSec * sampleRate), audioBuffer.length);
  const length = endSample - startSample;
  
  const channelData = audioBuffer.getChannelData(0);
  const pcmBytes = new Uint8Array(length * 2);
  const view = new DataView(pcmBytes.buffer);
  
  for (let i = 0; i < length; i++) {
    let sample = channelData[startSample + i];
    if (sample > 1.0) sample = 1.0;
    else if (sample < -1.0) sample = -1.0;
    const pcmSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
    view.setInt16(i * 2, pcmSample, true);
  }
  
  return pcmToWav(pcmBytes, sampleRate);
};

// JS 静音检测算法：基于滑动窗口 RMS
const detectSilencePoints = (
  channelData: Float32Array,
  sampleRate: number,
  dbThreshold: number = -35,
  minSilenceDuration: number = 0.6
): number[] => {
  const threshold = Math.pow(10, dbThreshold / 20); // 线性幅值阈值
  const windowSize = Math.floor(sampleRate * 0.02); // 20ms 窗宽
  const stepSize = Math.floor(sampleRate * 0.01); // 10ms 步长
  
  const silentBlocks: boolean[] = [];
  const blockTimes: number[] = [];
  
  for (let i = 0; i < channelData.length - windowSize; i += stepSize) {
    let sum = 0;
    for (let j = 0; j < windowSize; j++) {
      sum += channelData[i + j] * channelData[i + j];
    }
    const rms = Math.sqrt(sum / windowSize);
    silentBlocks.push(rms < threshold);
    blockTimes.push(i / sampleRate);
  }
  
  const minSilentBlocksCount = Math.ceil(minSilenceDuration / (stepSize / sampleRate));
  const silenceIntervals: { start: number; end: number }[] = [];
  let inSilence = false;
  let silenceStartBlock = 0;
  
  for (let b = 0; b < silentBlocks.length; b++) {
    if (silentBlocks[b]) {
      if (!inSilence) {
        inSilence = true;
        silenceStartBlock = b;
      }
    } else {
      if (inSilence) {
        inSilence = false;
        const durationBlocks = b - silenceStartBlock;
        if (durationBlocks >= minSilentBlocksCount) {
          silenceIntervals.push({
            start: blockTimes[silenceStartBlock],
            end: blockTimes[b - 1]
          });
        }
      }
    }
  }
  
  if (inSilence) {
    const durationBlocks = silentBlocks.length - silenceStartBlock;
    if (durationBlocks >= minSilentBlocksCount) {
      silenceIntervals.push({
        start: blockTimes[silenceStartBlock],
        end: blockTimes[silentBlocks.length - 1]
      });
    }
  }
  
  // 切割点取静音区间的正中心
  return silenceIntervals.map(interval => (interval.start + interval.end) / 2);
};

// ==========================================
// 4. UI 配色预设
// ==========================================
const PREMIUM_COLORS = [
  '#3b82f6', '#8b5cf6', '#ec4899', '#f43f5e', '#f59e0b', 
  '#10b981', '#06b6d4', '#6366f1', '#d946ef', '#14b8a6'
];

const emotionDescriptions: Record<string, string> = {
  calm: "温和、平静且自然",
  angry: "极度愤怒、咆哮且充满张力",
  sad: "低沉、哀伤且委屈",
  fear: "颤抖、恐惧且慌张",
  fearful: "颤抖、恐惧且慌张",
  happy: "高亢、喜悦且兴奋",
  tense: "紧绷、紧张且局促",
  // 补全提示词 emotion_class 枚举中存在但此处缺失的 4 种情绪，
  // 缺失时它们会退化为"温和、平静且自然"的默认朗读
  excited: "高亢、激动且充满张力",
  solemn: "庄重、严肃且沉稳",
  whisper: "气声、低语且刻意压低音量",
  neutral: "平静、自然且克制，不带明显感情色彩"
};

const getSpeakerColor = (speakerName: string, cast: CastMember[]) => {
  if (speakerName === '旁白' || speakerName === 'Narrator' || speakerName === 'NARRATOR') return 'rgba(148, 163, 184, 0.15)';
  const member = cast.find(c => 
    c.character_name === speakerName || 
    c.speaker_id === speakerName ||
    c.character_name_en === speakerName ||
    c.speaker_alias === speakerName ||
    (c.aliases && c.aliases.includes(speakerName))
  );
  if (member && member.colorHex) return member.colorHex;
  
  let hash = 0;
  for (let i = 0; i < speakerName.length; i++) hash = speakerName.charCodeAt(i) + ((hash << 5) - hash);
  return PREMIUM_COLORS[Math.abs(hash) % PREMIUM_COLORS.length];
};

const DEFAULT_TAGGING_SYSTEM_PROMPT = `<system_instruction>
你是一个没有任何废话、仅作为代码函数运行的“有声书剧本标注引擎”。
下方输入区已直接注入了【文件1】小说原文块 和 【文件2】角色表 cast.json。

核心任务：
  读取 cast.json，将【文件1】小说原文逐句拆解并标注为结构化 JSON 剧本（novel_script.json）。
  该 JSON 将直接驱动底层代码。

终极输出指令（极其重要）：
  1. 绝对禁止输出任何寒暄、确认语或诸如“好的，我将严格遵守”、“由于您尚未提供...”之类的废话。
  2. 绝对禁止进行“举例”或“演示”。
  3. 你唯一的输出必须是、且只能是一段符合 <output_schema> 的合法 JSON 字符串。任何其它文字都会导致系统崩溃！
</system_instruction>

<cast_json_reading_rules>
在开始标注前，必须先从 cast.json 中读取并内部记忆以下内容：

1. novel_name             → 填入输出 JSON 的 novel_name 字段
2. tts_model              → 填入输出 JSON 的 tts_model 字段
3. tagging_model          → 填入输出 JSON 的 tagging_model 字段
4. 每个 cast 条目必须读取以下六个字段，构建角色识别表：
   - character_name（中文主名）
   - character_name_en（英文名）
   - aliases（所有别称数组）
   - speaker_id
   - speaker_alias
   - assigned_voice_id（该角色对应的 TTS 音色，必须原样复制到每个 segment 的 voice 字段）

   原文中凡出现 character_name、character_name_en、aliases 中的任意一项，
   均视为同一角色，对应 segment 的以下字段统一填写：
   - char         = 该角色的 character_name（中文主名）
   - speaker_id   = 该角色的 speaker_id
   - speaker_alias = 该角色的 speaker_alias
   - voice        = 该角色的 assigned_voice_id（必须与 cast.json 完全一致，不得自行填写）

5. 旁白的 character_name 固定为 cast.json 中 speaker_id = "NARRATOR" 的那条。
   若 cast.json 无 NARRATOR 条目，则旁白统一填"旁白"。
   旁白的 voice 同样来自其 assigned_voice_id，不得与其他角色混用。
</cast_json_reading_rules>

<critical_rules>
【规则1】text 字段必须与原文完全一致，不得增删改写任何文字。

【规则2】char 字段识别规则（中文小说四种对话句式，必须全部覆盖）：

  基本原则：
  - 引号（"" 或 ""）内的内容       → char = 说话角色的 character_name
  - 引号外的所有内容（叙述/描写/
    环境/心理活动/归属标签）        → char = 旁白的 character_name
  - 角色名必须严格来自 cast.json 的 character_name，不得自创新名字

  四种句式的强制拆分规则：

  句式A，前置归属标签加对话：
    原文：克雷文先生说："你可以出去了。"
    segment 1: type=narration, char=旁白,   voice=旁白的voice, text="克雷文先生说："
    segment 2: type=dialogue,  char=克雷文, voice=克雷文的voice, text=""你可以出去了。""

  句式B，对话加后置归属标签（最常见误判点，必须严格执行）：
    原文："你可以出去了，"克雷文先生说。
    segment 1: type=dialogue,  char=克雷文, voice=克雷文的voice, text=""你可以出去了，""
    segment 2: type=narration, char=旁白,   voice=旁白的voice,   text="克雷文先生说。"

    强制规则：
    凡出现【引号结束 + 人名 + 言语动词 + 句号】结构，
    人名及言语动词之后的全部内容必须归属旁白，
    绝对不允许归属该人名角色本人。

  句式C，对话加归属标签加继续对话：
    原文："你先出去，"克雷文说，"我待会儿叫你。"
    segment 1: type=dialogue,  char=克雷文, voice=克雷文的voice, text=""你先出去，""
    segment 2: type=narration, char=旁白,   voice=旁白的voice,   text="克雷文说，"
    segment 3: type=dialogue,  char=克雷文, voice=克雷文的voice, text=""我待会儿叫你。""

  句式D，无归属标签的隐式对话：
    原文：玛丽走进房间。"你是谁？"
    segment 1: type=narration, char=旁白,           voice=旁白的voice, text="玛丽走进房间。"
    segment 2: type=dialogue,  char=（上下文推断）, voice=推断角色的voice, text=""你是谁？""
    若上下文无法确定说话人，则 char=旁白，voice=旁白的voice，并标注 char_inferred=false

  言语归属动词参考（含以下词的句子必须归旁白）：
    说、问、答、道、喊、叫、哭道、笑道、吼道、低声道、轻声说、
    冷冷地说、笑着说、皱眉道、摇头说、点头答、回答道、补充道、
    继续说、沉声道、急忙说、缓缓说、低语道、高声道、叫道、嚷道、
    哼道、叹道、苦笑道、微笑道、皱眉说、转向XX说、望着XX道
    等所有言语类动词及其修饰结构

【规则3】chapter 字段：
  - 遇到章节标题时，该行 type = "chapter_title"
  - 此后所有 segment 的 chapter 填写该章节名，直到下一章节标题出现

【规则4】emotion_class 必须从以下枚举中选择（英文小写）：
  calm / happy / sad / angry / fearful / excited / solemn / whisper / neutral

【规则5】audio_tag 必须是完整英文自然语言描述句（不少于8个词），直接用于 TTS style prompt。
  正确示例："Speak slowly with a warm nostalgic tone, like recalling a distant memory."
  错误示例："[calm]"（禁止标签格式）

【规则6】seq 从 1 开始全文连续递增，不按章节重置。

【规则7】分段边界规则：
  - 章节标题              → 独立一个 segment
  - 引号内的完整对话      → 独立一个 segment
  - 引号外的归属标签      → 独立一个 segment，不与对话合并
  - 引号外的旁白叙述      → 每段不超过150字，以自然句子边界切割
</critical_rules>

<emotion_guide>
情绪判断参考：
  calm     → 平静叙述、日常对话、环境描写
  happy    → 欢乐、喜悦、轻松愉快
  sad      → 悲伤、离别、痛苦、死亡
  angry    → 愤怒、争吵、指责
  fearful  → 恐惧、紧张、危险
  excited  → 兴奋、激动、高度紧张的动作场景
  solemn   → 庄重、严肃、誓言、葬礼
  whisper  → 低语、私下交谈、秘密
  neutral  → 过渡句、说明性文字、无法归类
</emotion_guide>

<execution_steps>
步骤1【读取 cast.json】：
  从 cast.json 中提取 novel_name、source_author、tts_model、tagging_model，
  构建完整的角色识别表，记录每个角色的 character_name、aliases、speaker_id、
  speaker_alias、assigned_voice_id 的完整映射关系。

步骤2【全文扫描】：
  通读全部原文，识别所有章节边界，
  对照角色识别表逐一确认每句对话的说话人归属。

步骤3【逐句拆分标注】：
  按规则2的四种句式逐句切割，为每个 segment 填写全部字段。
  每遇到后置归属标签，必须将其单独拆出作为旁白 segment。
  每个 segment 的 voice 字段必须从角色识别表中查找 assigned_voice_id 填入，
  禁止所有 segment 使用相同的 voice 值。

步骤4【统计 total_segments】：
  完成所有 segment 后统计总数，填入 total_segments 字段。

步骤5【JSON 输出】：
  严格按 output_schema 输出，不多字段不少字段。
</execution_steps>

<output_schema>
{
  "novel_name":      "（从 cast.json 读取）",
  "source_author":   "（从 cast.json 读取）",
  "tagging_model":   "（从 cast.json 读取）",
  "tts_model":       "（从 cast.json 读取）",
  "total_segments":  0,
  "global_segments": [
    {
      "seq":            1,
      "chapter":        "（当前章节名）",
      "type":           "chapter_title 或 narration 或 dialogue",
      "char":           "（严格来自 cast.json 的 character_name）",
      "speaker_id":     "（严格来自 cast.json 的 speaker_id）",
      "speaker_alias":  "（严格来自 cast.json 的 speaker_alias）",
      "voice":          "（严格来自 cast.json 的 assigned_voice_id，不同角色必须不同）",
      "char_in_cast":   true,
      "char_inferred":  false,
      "emotion_class":  "（枚举值）",
      "audio_tag":      "（完整英文自然语言句，不少于8个词）",
      "text":           "（与原文完全一致）"
    }
  ]
}
</output_schema>

<self_check_instruction>
输出 JSON 前，在内部完成以下检查（不输出检查过程）：
1.  total_segments 与 global_segments 数组长度是否一致？
2.  seq 是否从 1 连续递增，无重复无跳号？
3.  所有 text 是否与原文完全一致，无任何改写？
4.  所有 type=dialogue 的 segment，其 text 是否以引号开头并以引号结尾？
    若某 dialogue 的 text 不含引号，立即重新判断归属。
5.  所有包含后置归属标签（XX说、XX道、XX问等）的 segment，
    type 是否为 narration，char 是否为旁白？若不是，立即修正。
6.  所有 dialogue 的 char 是否为具体角色名（而非旁白）？
7.  所有 char 是否严格来自 cast.json 的 character_name？
    未收录角色标注 char_in_cast=false。
8.  所有 audio_tag 是否为完整英文自然语言句（不少于8个词）？
9.  所有 emotion_class 是否来自规定枚举？
10. 所有 type 是否从三个枚举值中选择？
11. novel_name、source_author、tts_model、tagging_model 是否全部来自 cast.json？
12. 检查所有 segment 的 voice 字段：
    是否每个角色的 voice 都与 cast.json 中该角色的 assigned_voice_id 完全一致？
    是否存在不同角色共用同一个 voice 值的情况（旁白除外）？
    若发现任何 voice 填写错误或所有 segment 的 voice 相同，立即逐一修正。
全部通过后再输出 JSON。
</self_check_instruction>

### 输入数据
【文件2】cast.json 如下：
<cast_json>
{{cast_json}}
</cast_json>

【上文语境】（仅用于指代消解）：
<previous_text>
{{previous_text}}
</previous_text>

【文件1】当前需要标注的小说原文块：
<current_chunk>
{{current_chunk}}
</current_chunk>
`;


// ==========================================
// IndexedDB 存储辅助类
// ==========================================
class AudioLabDB {
  private dbName = 'audiolab-db';
  private version = 1;
  private db: IDBDatabase | null = null;

  async init(): Promise<IDBDatabase> {
    if (this.db) return this.db;
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);
      request.onupgradeneeded = (e) => {
        const db = request.result;
        if (!db.objectStoreNames.contains('state')) {
          db.createObjectStore('state');
        }
        if (!db.objectStoreNames.contains('audios')) {
          db.createObjectStore('audios');
        }
      };
      request.onsuccess = () => {
        this.db = request.result;
        resolve(request.result);
      };
      request.onerror = () => reject(request.error);
    });
  }

  async get(storeName: string, key: string): Promise<any> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readonly');
      const store = tx.objectStore(storeName);
      const req = store.get(key);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async set(storeName: string, key: string, val: any): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      const req = store.put(val, key);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  }

  async delete(storeName: string, key: string): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      const req = store.delete(key);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  }

  async clear(storeName: string): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      const req = store.clear();
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  }

  async getAllKeys(storeName: string): Promise<string[]> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readonly');
      const store = tx.objectStore(storeName);
      const req = store.getAllKeys();
      req.onsuccess = () => resolve(req.result as string[]);
      req.onerror = () => reject(req.error);
    });
  }
}

const db = new AudioLabDB();

export default function AudiobookWorkstation() {
  // ==========================================
  // 5. 状态管理 (State)
  // ==========================================
  const [apiKeysStr, setApiKeysStr] = useState<string>(localStorage.getItem('gemini_keys') || '');
  const [modelName, setModelName] = useState<string>(localStorage.getItem('gemini_modelName') || 'gemini-3.1-flash-lite');
  const [ttsModelName, setTtsModelName] = useState<string>(localStorage.getItem('gemini_ttsModelName') || 'gemini-3.1-flash-tts-preview');
  const [rawText, setRawText] = useState<string>('');
  const [chunkSize, setChunkSize] = useState<number>(Number(localStorage.getItem('chunkSize')) || 1000);
  const [novelFileName, setNovelFileName] = useState<string>(localStorage.getItem('novelFileName') || '');
  const [castJsonStr, setCastJsonStr] = useState<string>('');
  const [castData, setCastData] = useState<CastJson | null>(null);

  // 提示词模板状态
  const [taggingSystemPrompt, setTaggingSystemPrompt] = useState<string>(localStorage.getItem('taggingSystemPromptV6') || DEFAULT_TAGGING_SYSTEM_PROMPT);


  
  const [chunks, setChunks] = useState<ScriptChunk[]>([]);
  const [globalSegments, setGlobalSegments] = useState<TaggedSegment[]>([]);
  const [ttsBatches, setTtsBatches] = useState<TTSBatch[]>([]);
  
  // 最终的音频存储映射 (seq -> Blob / URL)
  const [globalSequencedAudios, setGlobalSequencedAudios] = useState<Map<number, Blob>>(new Map());
  const [globalSequencedUrls, setGlobalSequencedUrls] = useState<Map<number, string>>(new Map());
  
  // 合成状态
  const [masterAudioBlob, setMasterAudioBlob] = useState<Blob | null>(null);
  const [masterAudioUrl, setMasterAudioUrl] = useState<string | null>(null);
  
  // 运行与控制状态
  const [isRunning, setIsRunning] = useState(false);
  const [isRunningTts, setIsRunningTts] = useState(false);
  const [pipelinePhase, setPipelinePhase] = useState<'setup' | 'tagging' | 'tagged' | 'tts' | 'stitching'>('setup');
  
  const [logs, setLogs] = useState<{msg: string, type: 'info'|'success'|'error'|'warn', time: string}[]>([]);
  
  const isRunningRef = useRef(false);
  const isRunningTtsRef = useRef(false);
  const lastRequestTimeRef = useRef<number>(0);
  const keyStatesRef = useRef<KeyRuntimeState[]>(JSON.parse(localStorage.getItem('audiolab_key_states') || '[]'));
  const lastUsedKeyIndexRef = useRef<number>(-1);
    
  const saveKeyStates = () => {
    localStorage.setItem('audiolab_key_states', JSON.stringify(keyStatesRef.current));
  };
  
  // UI 导航切换
  const [currentPage, setCurrentPage] = useState<number>(1);
  const pageSize = 100;
  const [isDbLoading, setIsDbLoading] = useState<boolean>(true);
  const [isRightPanelCollapsed, setIsRightPanelCollapsed] = useState<boolean>(false);
  
  const [activeTab, setActiveTab] = useState<'setup' | 'cast'>('setup');
  const [workspaceTab, setWorkspaceTab] = useState<'canvas' | 'batches' | 'master'>('canvas');
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null);
  const [selectedChunkIds, setSelectedChunkIds] = useState<Set<string>>(new Set());
  const [selectedBatchId, setSelectedBatchId] = useState<string | null>(null);
  const [playingSeq, setPlayingSeq] = useState<number | null>(null);
  const [regeneratingSeq, setRegeneratingSeq] = useState<number | null>(null);
  
  const [projectId, setProjectId] = useState<string>(localStorage.getItem('audiolab_projectId') || '');
  const [projectList, setProjectList] = useState<ProjectMeta[]>([]);
  const [isProjectModalOpen, setIsProjectModalOpen] = useState<boolean>(false);

  const audioPlayersRef = useRef<{ [seq: number]: HTMLAudioElement }>({});

  const addLog = useCallback((msg: string, type: 'info'|'success'|'error'|'warn' = 'info') => {
    setLogs(prev => [{msg, type, time: new Date().toLocaleTimeString()}, ...prev].slice(0, 200));
  }, []);

  // 1. IndexedDB 自动还原与 Demo 兜底
  useEffect(() => {
    const restoreProject = async () => {
      try {
        let pList: ProjectMeta[] = await db.get('state', 'project_list');
        let currentProjectId = localStorage.getItem('audiolab_projectId');

        // Migration from old single-project state
        if (!pList) {
          const oldState = await db.get('state', 'appState');
          if (oldState) {
            const defaultId = 'default_project';
            pList = [{
              id: defaultId,
              name: oldState.novelFileName || '迁移的默认项目',
              createdAt: Date.now(),
              updatedAt: Date.now(),
              totalChunks: oldState.chunks ? oldState.chunks.length : 0
            }];
            await db.set('state', 'project_list', pList);
            await db.set('state', `project_${defaultId}`, oldState);
            
            // Migrate audios
            const audioKeys = await db.getAllKeys('audios');
            for (const key of audioKeys) {
              if (!isNaN(Number(key))) { // Old format key
                const blob = await db.get('audios', key);
                await db.set('audios', `${defaultId}_${key}`, blob);
                await db.delete('audios', key);
              }
            }
            addLog("系统升级：已将旧数据平滑迁移为默认项目", "info");
            currentProjectId = defaultId;
          } else {
            pList = [];
          }
        }

        setProjectList(pList || []);

        if (pList && pList.length > 0 && !currentProjectId) {
          currentProjectId = pList[0].id;
        }

        if (currentProjectId) {
          setProjectId(currentProjectId);
          localStorage.setItem('audiolab_projectId', currentProjectId);
          const savedState = await db.get('state', `project_${currentProjectId}`);
          
          let hasSavedData = false;
          if (savedState) {
            if (savedState.rawText) { setRawText(savedState.rawText); hasSavedData = true; }
            if (savedState.chunkSize) setChunkSize(savedState.chunkSize);
            if (savedState.castJsonStr) {
              setCastJsonStr(savedState.castJsonStr);
              try {
                const parsed = JSON.parse(savedState.castJsonStr);
                setCastData(parsed);
              } catch {}
            }
            if (savedState.chunks) setChunks(savedState.chunks);
            // blob: URL 无法跨页面会话存活，刷新后从持久化的 Blob 重建
            if (savedState.ttsBatches) setTtsBatches(savedState.ttsBatches.map((b: TTSBatch) => ({
              ...b,
              audioUrl: b.audioBlob ? URL.createObjectURL(b.audioBlob) : undefined,
              slicedAudios: b.slicedAudios?.map(s => ({ ...s, url: URL.createObjectURL(s.blob) })),
            })));
            if (savedState.globalSegments) setGlobalSegments(savedState.globalSegments);
            if (savedState.modelName) setModelName(savedState.modelName);
            if (savedState.ttsModelName) setTtsModelName(savedState.ttsModelName);
            if (savedState.pipelinePhase) setPipelinePhase(savedState.pipelinePhase);
            if (savedState.taggingSystemPromptV6) setTaggingSystemPrompt(savedState.taggingSystemPromptV6);
          }

          // Restore audios for current project
          const audioKeys = await db.getAllKeys('audios');
          const audiosMap = new Map();
          const urlsMap = new Map();
          
          for (const key of audioKeys) {
            if (key.startsWith(`${currentProjectId}_`)) {
              const seqStr = key.substring(currentProjectId.length + 1);
              const seq = Number(seqStr);
              if (!isNaN(seq)) {
                const blob = await db.get('audios', key);
                if (blob instanceof Blob) {
                  audiosMap.set(seq, blob);
                  urlsMap.set(seq, URL.createObjectURL(blob));
                }
              }
            }
          }
          
          if (audiosMap.size > 0) {
            setGlobalSequencedAudios(audiosMap);
            setGlobalSequencedUrls(urlsMap);
          }
          
          if (hasSavedData) {
            addLog("成功从 IndexedDB 还原当前项目进度！", "success");
          } else {
            if (pList.length === 0) await loadDemoData();
          }
        } else {
          await loadDemoData();
        }
      } catch (err) {
        console.error("Restoring project failed", err);
        addLog("载入本地项目失败：" + err.message, "error");
        await loadDemoData();
      } finally {
        setIsDbLoading(false);
      }
    };

    const loadDemoData = async () => {
      try {
        const castRes = await fetch('/黑骏马角色.txt').catch(() => null);
        if (castRes && castRes.ok) {
          const text = await castRes.text();
          if (!text.trim().startsWith('<')) {
            setCastJsonStr(prev => prev ? prev : text);
          }
        }
        
        const keyRes = await fetch('/AK.txt').catch(() => null);
        if (keyRes && keyRes.ok) {
          const text = await keyRes.text();
          if (!text.trim().startsWith('<')) {
            setApiKeysStr(prev => prev ? prev : text);
          }
        }
        
        const textRes = await fetch('/黑骏马.txt').catch(() => null);
        if (textRes && textRes.ok) {
           const fullText = await textRes.text();
           if (!fullText.trim().startsWith('<')) {
             setRawText(prev => {
               if (prev) return prev;
               setTimeout(() => addLog("已成功加载演示数据 (黑骏马)。", "info"), 0);
               return fullText.substring(0, 6000);
             });
           }
        }
      } catch (e) {
        console.error("Failed to load demo data", e);
      }
    };

    restoreProject();
  }, []);

  // 2. 自动保存状态到 IndexedDB
  useEffect(() => {
    if (isDbLoading || !projectId) return;
    const stateObj = {
      rawText,
      castJsonStr,
      chunks,
      ttsBatches,
      globalSegments,
      modelName,
      ttsModelName,
      pipelinePhase,
      taggingSystemPromptV6: taggingSystemPrompt,
      novelFileName
    };
    db.set('state', `project_${projectId}`, stateObj).catch(err => {
      console.error("Failed to save state to IndexedDB", err);
    });

    setProjectList(prev => {
      const idx = prev.findIndex(p => p.id === projectId);
      const now = Date.now();
      let newList = [...prev];
      
      let needsDbWrite = false;
      if (idx !== -1) {
        if (
          newList[idx].totalChunks !== chunks.length ||
          newList[idx].name !== novelFileName ||
          now - newList[idx].updatedAt > 10000 // throttled update
        ) {
          newList[idx] = { ...newList[idx], updatedAt: now, totalChunks: chunks.length, name: novelFileName || newList[idx].name };
          needsDbWrite = true;
        }
      } else {
        newList.push({
          id: projectId,
          name: novelFileName || '未命名任务',
          createdAt: now,
          updatedAt: now,
          totalChunks: chunks.length
        });
        needsDbWrite = true;
      }

      if (needsDbWrite) {
        db.set('state', 'project_list', newList).catch(e => console.error(e));
        return newList;
      }
      return prev;
    });

  }, [
    isDbLoading,
    projectId,
    novelFileName,
    rawText,
    castJsonStr,
    chunks,
    ttsBatches,
    globalSegments,
    modelName,
    ttsModelName,
    pipelinePhase,
    taggingSystemPrompt,
  ]);

  // 3. 跨工程用户偏好自动保存到 localStorage
  useEffect(() => {
    localStorage.setItem('gemini_keys', apiKeysStr);
  }, [apiKeysStr]);

  useEffect(() => {
    localStorage.setItem('chunkSize', String(chunkSize));
  }, [chunkSize]);

  useEffect(() => {
    localStorage.setItem('novelFileName', novelFileName);
  }, [novelFileName]);

  useEffect(() => {
    localStorage.setItem('gemini_modelName', modelName);
  }, [modelName]);

  useEffect(() => {
    localStorage.setItem('gemini_ttsModelName', ttsModelName);
  }, [ttsModelName]);

  useEffect(() => {
    localStorage.setItem('taggingSystemPromptV6', taggingSystemPrompt);
  }, [taggingSystemPrompt]);


  // ==========================================
  // 多项目历史任务管理核心逻辑
  // ==========================================
  // 统一释放当前工程持有的全部音频资源（ObjectURL 与播放器缓存），
  // 供新建/切换/清空工程时调用；blob URL 不 revoke 会钉住底层音频内存，长篇项目切换几次即累积数百 MB
  const releaseAllAudioResources = () => {
    Object.values(audioPlayersRef.current).forEach(p => p.pause());
    audioPlayersRef.current = {};
    ttsBatches.forEach(b => {
      if (b.audioUrl) URL.revokeObjectURL(b.audioUrl);
      if (b.slicedAudios) b.slicedAudios.forEach(s => URL.revokeObjectURL(s.url));
    });
    globalSequencedUrls.forEach(url => URL.revokeObjectURL(url));
    if (masterAudioUrl) URL.revokeObjectURL(masterAudioUrl);
  };

  const createNewProject = () => {
    const newId = `proj_${Date.now()}`;
    const now = Date.now();
    
    releaseAllAudioResources();
    
    setRawText('');
    setNovelFileName('');
    setCastJsonStr('');
    setCastData(null);
    setChunks([]);
    setGlobalSegments([]);
    setTtsBatches([]);
    setGlobalSequencedAudios(new Map());
    setGlobalSequencedUrls(new Map());
    setMasterAudioBlob(null);
    setMasterAudioUrl(null);
    setPipelinePhase('setup');
    setLogs([]);
    
    setProjectId(newId);
    localStorage.setItem('audiolab_projectId', newId);
    setIsProjectModalOpen(false);
    addLog("已创建一个全新的空白任务", "success");
  };

  const loadProject = async (targetId: string) => {
    if (targetId === projectId) {
      setIsProjectModalOpen(false);
      return;
    }
    
    setIsDbLoading(true);
    try {
      const savedState = await db.get('state', `project_${targetId}`);
      if (!savedState) {
        addLog("项目数据不存在", "error");
        return;
      }
      
      setProjectId(targetId);
      localStorage.setItem('audiolab_projectId', targetId);
      
      // 先释放上一个工程占用的音频资源，防止项目间来回切换累积泄漏
      releaseAllAudioResources();
      
      setRawText(savedState.rawText || '');
      setNovelFileName(savedState.novelFileName || '');
      setCastJsonStr(savedState.castJsonStr || '');
      try {
        setCastData(JSON.parse(savedState.castJsonStr));
      } catch { setCastData(null); }
      setChunks(savedState.chunks || []);
      // 持久化的 blob: URL 在页面刷新或项目切换后已失效，必须从 Blob 重新生成 URL
      setTtsBatches((savedState.ttsBatches || []).map((b: TTSBatch) => ({
        ...b,
        audioUrl: b.audioBlob ? URL.createObjectURL(b.audioBlob) : undefined,
        slicedAudios: b.slicedAudios?.map(s => ({ ...s, url: URL.createObjectURL(s.blob) })),
      })));
      setGlobalSegments(savedState.globalSegments || []);
      if (savedState.pipelinePhase) setPipelinePhase(savedState.pipelinePhase);
      if (savedState.chunkSize) setChunkSize(savedState.chunkSize);
      
      const audioKeys = await db.getAllKeys('audios');
      const audiosMap = new Map();
      const urlsMap = new Map();
      
      for (const key of audioKeys) {
        if (key.startsWith(`${targetId}_`)) {
          const seq = Number(key.substring(targetId.length + 1));
          if (!isNaN(seq)) {
            const blob = await db.get('audios', key);
            if (blob instanceof Blob) {
              audiosMap.set(seq, blob);
              urlsMap.set(seq, URL.createObjectURL(blob));
            }
          }
        }
      }
      
      setGlobalSequencedAudios(audiosMap);
      setGlobalSequencedUrls(urlsMap);
      
      setMasterAudioBlob(null);
      setMasterAudioUrl(null);
      setLogs([]);
      setIsProjectModalOpen(false);
      addLog(`成功载入项目: ${savedState.novelFileName || targetId}`, "success");
    } catch (err) {
      addLog("载入项目失败: " + err.message, "error");
    } finally {
      setIsDbLoading(false);
    }
  };

  const deleteProject = async (targetId: string) => {
    if (!window.confirm("确定要删除此项目吗？该项目的所有状态及生成的TTS音频将被永久清除，且无法恢复。")) {
      return;
    }
    
    try {
      await db.delete('state', `project_${targetId}`);
      
      const newList = projectList.filter(p => p.id !== targetId);
      await db.set('state', 'project_list', newList);
      setProjectList(newList);
      
      const audioKeys = await db.getAllKeys('audios');
      for (const key of audioKeys) {
        if (key.startsWith(`${targetId}_`)) {
          await db.delete('audios', key);
        }
      }
      
      addLog("项目删除成功", "success");
      
      if (targetId === projectId) {
        if (newList.length > 0) {
          await loadProject(newList[0].id);
        } else {
          createNewProject();
        }
      }
    } catch (err) {
      addLog("删除项目失败: " + err.message, "error");
    }
  };

  // ==========================================
  // 6. 核心处理：智能切片与编码读取
  // ==========================================
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setNovelFileName(file.name.replace(/\.[^/.]+$/, ""));
    addLog(`Reading file: ${file.name} (${(file.size / 1024).toFixed(1)} KB)...`, "info");
    const reader = new FileReader();
    
    reader.onload = async (evt) => {
      const buffer = new Uint8Array(evt.target?.result as ArrayBuffer);
      const isUtf = isUtf8(buffer);
      const encoding = isUtf ? 'utf-8' : 'gbk';
      
      const decoder = new TextDecoder(encoding);
      const text = decoder.decode(buffer);
      setRawText(text);
      addLog(`File read successfully using [${encoding.toUpperCase()}] encoding. Text length: ${text.length} chars.`, "success");
    };
    
    reader.readAsArrayBuffer(file);
  };

  // 生成 Layer 1 切片
  const handleParseAndChunk = () => {
    try {
      if (!castJsonStr.trim()) {
        addLog("请配置 Cast JSON 以进行映射", "error");
        return;
      }
      
      const parsedCast = JSON.parse(castJsonStr) as CastJson;
      parsedCast.cast = parsedCast.cast.map((c, i) => ({
        ...c,
        colorHex: PREMIUM_COLORS[i % PREMIUM_COLORS.length]
      }));
      setCastData(parsedCast);
      localStorage.setItem('gemini_keys', apiKeysStr);
      
      addLog("Layer 1: Initializing smart chunker...", "info");
      const chunkData = smartChunking(rawText, chunkSize, 200);
      
      const newChunks: ScriptChunk[] = chunkData.map((data, idx) => ({
        id: `chunk-${Date.now()}-${idx}`,
        index: idx,
        originalText: data.originalText,
        previousTextContext: data.previousTextContext,
        status: 'pending',
        segments: []
      }));
      
      setChunks(newChunks);
      setPipelinePhase('tagging');
      setWorkspaceTab('canvas');
      setIsRightPanelCollapsed(true);
      addLog(`Layer 1: Smart chunking completed. Generated ${newChunks.length} chunks. Ready for LLM tagging.`, "success");
      setActiveTab('cast');
    } catch (e) {
      addLog(`解析 Cast JSON 失败: ${(e as Error).message}`, "error");
    }
  };

  // ==========================================
  // 7. KeyRotator 调度机制
  // ==========================================
  const getAvailableKey = async (isManual = false): Promise<string | null> => {
    const keys = apiKeysStr.split('\n').map(k => k.trim()).filter(k => k);
    if (keys.length === 0) return null;
    
    if (keyStatesRef.current.length !== keys.length) {
      keyStatesRef.current = keys.map(key => ({ key, isRateLimited: false, cooldownUntil: 0 }));
    }
    
    const scanOnce = (): string | null => {
      const now = Date.now();
      let foundKey: KeyRuntimeState | null = null;
      let startIndex = lastUsedKeyIndexRef.current + 1;
      
      for(let i = 0; i < keys.length; i++) {
         let checkIndex = (startIndex + i) % keys.length;
         let keyState = keyStatesRef.current[checkIndex];
         
         if (!keyState.isRateLimited || now > keyState.cooldownUntil) {
            keyState.isRateLimited = false;
            keyState.cooldownUntil = 0;
            foundKey = keyState;
            lastUsedKeyIndexRef.current = checkIndex;
            break;
         }
      }
      saveKeyStates();
      
      return foundKey ? foundKey.key : null;
    };
    
    // 手动操作（单句重合成 / Level 3 逐句合成）在流水线未运行时也必须能立即拿到可用 Key
    const immediateKey = scanOnce();
    if (immediateKey) return immediateKey;
    
    while (isRunningRef.current || isRunningTtsRef.current || isManual) {
      const found = scanOnce();
      if (found) return found;
      
      const now = Date.now();
      const earliestKey = [...keyStatesRef.current].sort((a, b) => a.cooldownUntil - b.cooldownUntil)[0];
      const waitTime = Math.max(1000, earliestKey.cooldownUntil - now);
      addLog(`等待 ${Math.round(waitTime/1000)}秒以解除 429 限流...`, "warn");
      await new Promise(r => setTimeout(r, waitTime));
    }
    return null;
  };

  const waitRateLimitDelay = async () => {
    const now = Date.now();
    const elapsed = now - lastRequestTimeRef.current;
    const delay = 2000 - elapsed; // 强制每次调用间隔至少 2 秒，防429
    if (delay > 0) {
      await new Promise(r => setTimeout(r, delay));
    }
    lastRequestTimeRef.current = Date.now();
  };

  // ==========================================
  // 8. Layer 2: LLM 打标引擎 (KeyRotator 轮询)
  // ==========================================
  // 自动从文本中提取并分析角色，并生成 Cast JSON

  const callGeminiFlash = async (chunkText: string, previousContext: string, apiKey: string, castDataArg: CastJson): Promise<TaggedSegment[]> => {
    await waitRateLimitDelay();
    
    let cleanPrompt = taggingSystemPrompt;
    
    // 【核心修复1】：彻底切除 System Prompt 中可能残留的【输入数据】空壳区域，防止模型产生“缺少数据”的幻觉
    if (cleanPrompt.includes("### 输入数据")) {
      cleanPrompt = cleanPrompt.split("### 输入数据")[0].trim();
    }
      
    // 动态擦除可能导致幻觉的负面约束
    cleanPrompt = cleanPrompt.replace(/绝对禁止.*由于您尚未提供.*之类的废话。/g, "绝对禁止输出任何寒暄、确认语。直接输出纯 JSON！");
      
    // 分离指令与数据：将规则作为原生 System Instruction，将数据作为 User Message
    const dataPrompt = `### 输入数据
【文件2】cast.json 如下：
<cast_json>
${JSON.stringify(castDataArg || {}, null, 2)}
</cast_json>

【上文语境】（仅用于指代消解）：
<previous_text>
${previousContext || "无"}
</previous_text>

【文件1】当前需要标注的小说原文块：
<current_chunk>
${chunkText}
</current_chunk>

🚨**立刻执行任务**🚨：
我已经提供了真实的【小说原文】和完整的【cast.json】。
请立即处理上述 <current_chunk> 中的真实文本，不要生成测试示例，也不要抱怨缺失文件。
直接输出合法的纯 JSON 数据！
`;

    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${apiKey}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: cleanPrompt }] },
        contents: [{ role: 'user', parts: [{ text: dataPrompt }] }],
        generationConfig: {
          temperature: modelName.toLowerCase().startsWith("gemini-3") ? 1.0 : 0.1,
          // 【核心修复2】：强行约束 API 只能返回 JSON 格式数据，从物理上切断模型输出寒暄废话的可能
          responseMimeType: "application/json"
        }
      })
    });

    if (res.status === 429) {
      const keyState = keyStatesRef.current.find(k => k.key === apiKey);
      if (keyState) {
        keyState.isRateLimited = true;
        keyState.cooldownUntil = Date.now() + 60000; // 冷却 60 秒
        saveKeyStates();
      }
      throw new Error("429_RATE_LIMIT");
    }

    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      let errMsg = `HTTP ${res.status}`;
      try {
        const errJson = JSON.parse(errText);
        if (errJson.error && errJson.error.message) {
          errMsg += `: ${errJson.error.message}`;
        }
      } catch (e) {
        errMsg += ` ${errText.slice(0, 50)}`;
      }
      throw new Error(errMsg);
    }

    const data = await res.json();
    const resultText = data.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!resultText) throw new Error("Empty Response");

    let cleaned = resultText.trim();
    const jsonMatch = cleaned.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    if (jsonMatch) {
      cleaned = jsonMatch[1].trim();
    } else {
      const braceMatch = cleaned.match(/\{[\s\S]*\}/);
      if (braceMatch) {
        cleaned = braceMatch[0];
      }
    }

    // ========== 双格式兼容解析 ==========
    // 优先尝试 JSON 格式（新提示词输出）
    try {
      const parsed = JSON.parse(cleaned);
      const segments = parsed.global_segments || [];
      if (segments.length > 0) {
        addLog(`[Parser] ✅ 成功解析 JSON 格式，共 ${segments.length} 个 segment`, "success");
        return segments.map((seg: any) => ({
          seq: seg.seq,
          chapter: seg.chapter,
          type: seg.type,
          char: seg.char || "旁白",
          speaker_id: seg.speaker_id,
          speaker_alias: seg.speaker_alias,
          voice: seg.voice || castDataArg?.cast.find(c => c.character_name === seg.char || c.speaker_alias === seg.char)?.assigned_voice_id || 
                 castDataArg?.cast.find(c => c.speaker_id === "NARRATOR")?.assigned_voice_id || "Aoede",
          char_in_cast: seg.char_in_cast,
          char_inferred: seg.char_inferred,
          emotion_class: seg.emotion_class || "calm",
          audio_tag: seg.audio_tag || "",
          text: seg.text || ""
        }));
      }
    } catch {
      // JSON 解析失败，继续尝试旧的 [SPEAKER: ...] 格式
    }

    // 回退到旧的 [SPEAKER: ...] 行格式解析
    const lines = cleaned.split('\n');
    const segments: TaggedSegment[] = [];
    const lineRegex = /^\[SPEAKER:\s*(.*?),\s*voice="(.*?)",\s*emotion="(.*?)",\s*style="(.*?)"\]\s*(.*)$/;

    lines.forEach(line => {
      const trimmed = line.trim();
      if (!trimmed) return;
      
      const match = trimmed.match(lineRegex);
      if (match) {
        const [, speaker, voice, emotion, style, text] = match;
        
        let emotionClass = emotion.trim().toLowerCase();
        if (emotionClass === 'fear') emotionClass = 'fearful';
        if (emotionClass === 'tense') emotionClass = 'excited';
        
        segments.push({
          char: speaker.trim(),
          voice: voice.trim(),
          emotion_class: emotionClass,
          audio_tag: style.trim(),
          text: text.trim()
        });
      } else {
        if (trimmed.length > 0 && !trimmed.startsWith('[')) {
          segments.push({
            char: "旁白",
            voice: castDataArg?.cast.find(c => c.character_name === "旁白" || c.speaker_id === "NARRATOR")?.assigned_voice_id || "Aoede",
            emotion_class: "calm",
            audio_tag: "Calm narration with steady pacing",
            text: trimmed
          });
        }
      }
    });

    if (segments.length > 0) {
      addLog(`[Parser] ✅ 使用 [SPEAKER] 格式解析，共 ${segments.length} 个 segment`, "info");
      return segments;
    }

    throw new Error(`无法解析模型输出（既非 JSON 也非 SPEAKER 格式）\nOutput preview: ${cleaned.substring(0, 150)}`);
  };

  const processAllChunks = async (resume: boolean = false) => {
    setIsRunning(true);
    isRunningRef.current = true;
    
    // 动态解析当前的 castJsonStr 以防用户修改了文本框但没点 L1 按钮导致不同步
    let currentCastData = castData;
    if (castJsonStr.trim()) {
      try {
        currentCastData = JSON.parse(castJsonStr);
        setCastData(currentCastData);
      } catch (e) {
        addLog("❌ Cast JSON 格式解析失败！请检查拼写、逗号或括号。", "error");
        setIsRunning(false);
        return;
      }
    }
    
    if (!currentCastData || !currentCastData.cast || currentCastData.cast.length === 0) {
      addLog("❌ Cast JSON 配置错误或缺少 cast 数组！请检查右侧面板的 JSON 格式。", "error");
      setIsRunning(false);
      return;
    }
    
    const keys = apiKeysStr.split('\n').map(k => k.trim()).filter(k => k);
    if (keys.length === 0) {
      addLog("未配置 API Key", "error");
      setIsRunning(false);
      return;
    }
    
    let activeChunks = resume 
      ? [...chunks] 
      : chunks.map(c => ({ ...c, status: undefined, segments: [] }));
    
    if (!resume) {
      setChunks(activeChunks);
    }
    
    addLog(`Layer 2: Starting LLM tagging loop with ${keys.length} key(s)...`, "info");

    for (let i = 0; i < activeChunks.length; i++) {
      if (!isRunningRef.current) break;
      let currentChunk = activeChunks[i];
      if (currentChunk.status === 'success') continue;

      setChunks(prev => {
        const next = [...prev];
        const idx = next.findIndex(c => c.id === currentChunk.id);
        if (idx !== -1) next[idx].status = 'processing';
        return next;
      });
      // 取消 setSelectedChunkId 跟踪，防止人工查看时乱跳

      try {
        const key = await getAvailableKey();
        if (!key) throw new Error("Key loop stopped or exhausted.");

        addLog(`Tagging Chunk [${i + 1}/${activeChunks.length}] using key [${key.substring(0,8)}...]`, "info");
        const segments = await callGeminiFlash(currentChunk.originalText, currentChunk.previousTextContext, key, currentCastData!);
        
        // 🚨 铁律校验：绝对保真度检查
        const joinedText = segments.map(s => s.text).join('');
        const origClean = currentChunk.originalText.replace(/\s+/g, '');
        const joinedClean = joinedText.replace(/\s+/g, '');
        
        if (Math.abs(origClean.length - joinedClean.length) > Math.max(15, origClean.length * 0.15)) {
          addLog(`Chunk [${i + 1}] 保真警告: 原文 ${origClean.length} 字符, 输出 ${joinedClean.length} 字符。请在右侧人工核对。`, "warn");
          // 改为警告而不是直接失败，让用户可以预览并手动编辑
        } else if (segments.length === 0) {
          throw new Error("模型未输出任何有效剧本标签");
        }

        // 校验 2: 对越界说话人做 Fallback 映射（用当次解析的 currentCastData 而非渲染闭包里的 castData，
        // 长流水线运行期间用户改动 Cast JSON 时避免新旧角色表混用）
        const validSpeakers = ['旁白', 'Narrator', 'NARRATOR', ...(currentCastData?.cast.flatMap(c => [c.character_name, c.speaker_id, c.character_name_en, c.speaker_alias, ...(c.aliases || [])].filter(Boolean) as string[]) || [])];
        const unknownSet = new Set<string>();
        segments.forEach(s => {
          if (!validSpeakers.includes(s.char)) {
            if (!unknownSet.has(s.char)) {
              addLog(`警告: 未知说话人 "${s.char}"。自动修正为旁白`, "warn");
              unknownSet.add(s.char);
            }
            s.char = '旁白';
            s.voice = currentCastData?.cast.find(c => c.character_name === '旁白' || c.speaker_id === 'NARRATOR')?.assigned_voice_id || 'Aoede';
          }
        });

        setChunks(prev => {
          const next = [...prev];
          const idx = next.findIndex(c => c.id === currentChunk.id);
          if (idx !== -1) {
            next[idx].status = 'success';
            next[idx].segments = segments;
          }
          return next;
        });
        addLog(`✅ Chunk [${i + 1}] success: ${segments.length} segments tagged.`, "success");

      } catch (error: any) {
        if (error.message === "429_RATE_LIMIT" || error.message.startsWith("HTTP 5")) {
          i--; // 重新处理此块
          const errorType = error.message === "429_RATE_LIMIT" ? "429 限流" : error.message;
          addLog(`触发 ${errorType}，自动切换下一个 Key 重试...`, "warn");
          
          // 如果是 5xx 错误，给当前的 Key 也加一点短冷却，防止连续轰炸崩溃的节点
          if (error.message.startsWith("HTTP 5")) {
            const currentKey = keyStatesRef.current[lastUsedKeyIndexRef.current]?.key;
            if (currentKey) {
              const state = keyStatesRef.current.find(k => k.key === currentKey);
              if (state) {
                state.isRateLimited = true;
                state.cooldownUntil = Date.now() + 10000; // 5xx 给予 10 秒短冷却
                saveKeyStates();
              }
            }
          }
          continue;
        }

        addLog(`❌ Chunk [${i + 1}] 失败: ${error.message}。`, "error");
        
        setChunks(prev => {
          const next = [...prev];
          const idx = next.findIndex(c => c.id === currentChunk.id);
          if (idx !== -1) next[idx].status = 'failed';
          return next;
        });
      }
    }

    setIsRunning(false);
    isRunningRef.current = false;
    
    // 如果全部成功，自动完成全局 sequence 排序并生成聚合批次 (Layer 2.5)
    const allSuccessful = activeChunks.every(c => c.status === 'success');
    if (allSuccessful) {
      addLog("🎉 所有切片打标成功！正在执行 Layer 2.5 情绪聚合...", "success");
      setPipelinePhase('tagged');
      setWorkspaceTab('batches');
      buildTTSBatches(activeChunks);
    } else {
      addLog("⚠️ 打标流水线完成，但存在失败段落，请勾选补扫。", "warn");
    }
  };

    const handleSelectFailed = () => {
    const failedIds = chunks.filter(c => c.status === 'failed' || c.status === 'rescued').map(c => c.id);
    setSelectedChunkIds(new Set(failedIds));
    };

    const handleRetrySelectedChunks = async () => {
    if (selectedChunkIds.size === 0) return addLog("请先勾选需要补扫的分块", "warn");
    
    let currentCastData = castData;
    if (castJsonStr.trim()) {
      try {
        currentCastData = JSON.parse(castJsonStr);
      } catch {
        addLog("❌ Cast JSON 格式解析失败！请检查拼写、逗号或括号。", "error");
        return;
      }
    }
    if (!currentCastData || !currentCastData.cast || currentCastData.cast.length === 0) {
      addLog("❌ Cast JSON 配置错误或缺少 cast 数组！", "error");
      return;
    }
    
    addLog(`启动人工补扫，共 ${selectedChunkIds.size} 块...`, "info");
      
    for (const chunkId of Array.from(selectedChunkIds)) {
      const chunkIdx = chunks.findIndex(c => c.id === chunkId);
      if (chunkIdx === -1) continue;
        
      setChunks(prev => prev.map(c => c.id === chunkId ? { ...c, status: 'processing' } : c));
        
      try {
        const key = await getAvailableKey(true);
        if (!key) throw new Error("无可用 Key");
          
        const segments = await callGeminiFlash(chunks[chunkIdx].originalText, chunks[chunkIdx].previousTextContext, key, currentCastData);
          
        setChunks(prev => prev.map(c => c.id === chunkId ? { ...c, status: 'success', segments } : c));
        setSelectedChunkIds(prev => {
          const next = new Set(prev);
          next.delete(chunkId);
          return next;
        });
        addLog(`✅ 补扫 [分块 ${chunks[chunkIdx].index + 1}] 成功`, "success");
      } catch (error: any) {
        addLog(`❌ 补扫 [分块 ${chunks[chunkIdx].index + 1}] 失败: ${error.message}`, "error");
        setChunks(prev => prev.map(c => c.id === chunkId ? { ...c, status: 'failed' } : c));
      }
    }
    };

  // ==========================================
  // 9. Layer 2.5: 情绪批量聚合器
  // ==========================================
  const buildTTSBatches = (completedChunks: ScriptChunk[]) => {
    const TAG_TO_INLINE: Record<string, string> = {
      "calm": "", "warm": "", "narration": "", "whisper": "[whisper]",
      "excitedly": "[excitedly]", "long pause": "[long pause]",
      "short pause": "[short pause]", "fast": "[fast]", "slow": "[slow]",
      "angry": "[angry]", "sad": "[sadly]", "cheerful": "[cheerfully]",
      "gentle pace": "[slowly]", "reflective": "", "dignified": "", "introspective": ""
    };

    const byteLen = (text: string) => new Blob([text]).size;
    const cleanText = (text: string) => text.replace(/^#{1,6}\s+/, "").replace(/\*\*/g, "").replace(/["""「」『』]/g, "").trim();
    const isChapterTitle = (text: string) => /^#{1,6}\s/.test(text.trim()) || /^第[零一二三四五六七八九十百千\d]+章/.test(text.trim());
    const extractChapter = (text: string) => { const m = text.match(/第[零一二三四五六七八九十百千\d]+章/); return m ? m[0] : "未标注章节"; };
    const tagsToInlinePrefix = (tags?: string[]) => {
      if (!tags || tags.length === 0) return "";
      return tags.map(t => TAG_TO_INLINE[t.toLowerCase()] || "").filter(Boolean).join(" ");
    };

    const charLookup = new Map<string, CastMember>();
    if (castData) {
      for (const entry of castData.cast) {
        const keys = [entry.character_name, entry.character_name_en, entry.speaker_id, entry.speaker_alias, ...(entry.aliases || [])].filter(Boolean) as string[];
        for (const key of keys) {
          if (!charLookup.has(key)) charLookup.set(key, entry);
        }
      }
    }

    // 1. 全局按顺序赋予递增 seq 序号并 enrich
    let seqCounter = 1;
    const allSegs: any[] = [];
    
    completedChunks.forEach(chunk => {
      chunk.segments.forEach(seg => {
        const titleFlag = isChapterTitle(seg.text);
        const cleaned = titleFlag ? cleanText(seg.text).replace(/^第/, "第").replace(/章$/, "章，") : cleanText(seg.text);
        const charInfo = charLookup.get(seg.char);
        const entry = charInfo || (castData?.cast[0] || {} as CastMember);

        const inlinePrefix = tagsToInlinePrefix(
          seg.emotion_class === "calm" ? entry.tags_suggestion :
          seg.emotion_class === "angry" ? ["angry"] :
          seg.emotion_class === "sad" ? ["sad"] :
          seg.emotion_class === "excited" ? ["excitedly"] :
          seg.emotion_class === "whisper" ? ["whisper"] :
          entry.tags_suggestion
        );

        const ttsText = inlinePrefix ? `${inlinePrefix} ${cleaned}` : cleaned;

        allSegs.push({
          ...seg,
          // LLM 返回的 seq 只是分块内局部编号，必须无条件全局重编号
          // 否则跨分块 seq 冲突会覆盖音频缓存并打乱时间轴，整书拼接必然失败
          seq: seqCounter++,
          chapter: seg.chapter || extractChapter(seg.text),
          type: seg.type || (titleFlag ? "chapter_title" : "content"),
          speaker_id: seg.speaker_id || entry.speaker_id,
          speaker_alias: seg.speaker_alias || entry.speaker_alias || seg.char,
          voice_profile: entry.voice_profile,
          clean_text: cleaned,
          tts_text: ttsText,
          estimated_bytes: byteLen(ttsText)
        });
      });
    });
    
    setGlobalSegments(allSegs as TaggedSegment[]);

    // 2. 按 speaker_alias 分轨
    const tracks = new Map<string, any[]>();
    for (const seg of allSegs) {
      if (!tracks.has(seg.speaker_alias)) tracks.set(seg.speaker_alias, []);
      tracks.get(seg.speaker_alias)!.push(seg);
    }

    // 3. 构建批次
    const batches: TTSBatch[] = [];
    let batchCounter = 0;
    const BYTE_LIMIT = 3800;

    for (const [speakerAlias, segs] of tracks) {
      segs.sort((a: any, b: any) => a.seq - b.seq);
      let cur: any = null;

      for (const seg of segs) {
        const segBytes = seg.estimated_bytes + 2;
        const needNew = !cur || seg.emotion_class !== cur.emotionClass || cur.totalBytes + segBytes > BYTE_LIMIT;

        if (needNew) {
          if (cur && cur.lines.length > 0) {
            batchCounter++;
            const fullText = cur.lines.map((l: any) => {
              const t = l.text.trimEnd();
              return /[。！？…～]$/.test(t) ? t : t + "。";
            }).join("\n");
            batches.push({
              id: `${cur.speakerAlias}_B${batchCounter}`,
              speaker: cur.speakerAlias,
              voiceId: cur.voiceId,
              emotionClass: cur.emotionClass,
              text: fullText,
              seqs: cur.lines.map((l: any) => l.seq),
              status: 'pending',
              ttsPrompt: cur.voiceProfile,
              totalBytes: cur.totalBytes,
              anchorMap: cur.lines.map((l: any, idx: number) => ({
                line_index: idx, seq: l.seq, char: l.char, chapter: l.chapter, preview: l.rawText.slice(0, 20)
              }))
            });
          }
          cur = {
            speakerAlias,
            voiceId: seg.voice,
            voiceProfile: seg.voice_profile,
            emotionClass: seg.emotion_class,
            lines: [],
            totalBytes: 0
          };
        }

        cur.lines.push({
          seq: seg.seq, text: seg.tts_text, rawText: seg.clean_text, char: seg.char, chapter: seg.chapter
        });
        cur.totalBytes += segBytes;
      }

      if (cur && cur.lines.length > 0) {
        batchCounter++;
        const fullText = cur.lines.map((l: any) => {
          const t = l.text.trimEnd();
          return /[。！？…～]$/.test(t) ? t : t + "。";
        }).join("\n");
        batches.push({
          id: `${cur.speakerAlias}_B${batchCounter}`,
          speaker: cur.speakerAlias,
          voiceId: cur.voiceId,
          emotionClass: cur.emotionClass,
          text: fullText,
          seqs: cur.lines.map((l: any) => l.seq),
          status: 'pending',
          ttsPrompt: cur.voiceProfile,
          totalBytes: cur.totalBytes,
          anchorMap: cur.lines.map((l: any, idx: number) => ({
            line_index: idx, seq: l.seq, char: l.char, chapter: l.chapter, preview: l.rawText.slice(0, 20)
          }))
        });
      }
    }

    setTtsBatches(batches);
    addLog(`Layer 2.5: 聚合完成！共分配了 ${allSegs.length} 个全局对白，整理为 ${batches.length} 个情绪大批次。`, "success");
  };

  // ==========================================
  // 10. Layer 3: 单说话人 TTS 批量合成与重试
  // ==========================================
  const callGeminiTTS = async (text: string, voiceName: string, apiKey: string): Promise<Blob> => {
    await waitRateLimitDelay();
    
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${ttsModelName}:generateContent?key=${apiKey}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: text }] }],
        generationConfig: {
          temperature: 1.0,
          responseModalities: ["AUDIO"],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: {
                voiceName: voiceName
              }
            }
          }
        }
      })
    });

    if (res.status === 429) {
      const keyState = keyStatesRef.current.find(k => k.key === apiKey);
      if (keyState) {
        keyState.isRateLimited = true;
        keyState.cooldownUntil = Date.now() + 60000;
        saveKeyStates();
      }
      throw new Error("429_RATE_LIMIT");
    }

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    const part = data.candidates?.[0]?.content?.parts?.[0];
    if (part && part.inlineData) {
      const base64Data = part.inlineData.data;
      const mimeType = part.inlineData.mimeType || "audio/pcm";
      
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // 若输出为原始无头 PCM，前端将其重组封装为可播放的 WAV
      if (mimeType.includes("pcm") || mimeType.includes("L16")) {
        return pcmToWav(bytes, 24000);
      } else {
        return new Blob([bytes], { type: 'audio/wav' });
      }
    } else {
      throw new Error("API 未返回有效的多模态音频流");
    }
  };

  const callGeminiTTSWithRetry = async (text: string, voiceName: string, maxRetries: number = 3): Promise<Blob> => {
    let lastError: any = null;
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      if (!isRunningTtsRef.current) throw new Error("TTS Pipeline stopped by user.");
      
      const key = await getAvailableKey();
      if (!key) throw new Error("No API key available.");
      
      try {
        const wav = await callGeminiTTS(text, voiceName, key);
        return wav;
      } catch (err: any) {
        lastError = err;
        if (err.message === "429_RATE_LIMIT") {
          attempt--; // 同步重试此轮，不占重试计数
          continue;
        }
        
        addLog(`TTS Batch 生成第 ${attempt} 次失败: ${err.message}。正在执行指数退避重试...`, "warn");
        if (attempt < maxRetries) {
          // 指数退避：1s, 2s, 4s
          await new Promise(r => setTimeout(r, attempt * 1000));
        }
      }
    }
    throw lastError || new Error("TTS generation failed after retries.");
  };

  // ==========================================
  // 11. Layer 3.5: 音频切割校验器 (含三级降级 Fallback)
  // ==========================================
  const sliceAndValidateBatch = async (wavBlob: Blob, batch: TTSBatch): Promise<{ seq: number; blob: Blob; url: string; fallbackLevel?: number }[]> => {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    let audioBuffer: AudioBuffer;
    try {
      const arrayBuffer = await wavBlob.arrayBuffer();
      audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    } finally {
      // 浏览器并发 AudioContext 有硬上限（约 6 个），用完必须关闭，否则长篇多批次后 decodeAudioData 抛异常中断流水线
      audioContext.close().catch(() => {});
    }
    const channelData = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;
    
    const expectedCount = batch.seqs.length;
    
    // 【Level 1: 默认静音切割阈值测试】
    let splitPoints = detectSilencePoints(channelData, sampleRate, -35, 0.6);
    let slicesCount = splitPoints.length + 1;
    let fallbackLevel = 0; // 完美匹配
    
    if (slicesCount !== expectedCount) {
      addLog(`[Batch ${batch.id}] 默认参数切出 ${slicesCount} 段 (预期为 ${expectedCount} 段)。触发一级降级：动态调整静音阈值...`, "warn");
      // 一级降级 retries 配置
      const fallbackThresholds = [
        { db: -30, dur: 0.5 },
        { db: -40, dur: 0.7 },
        { db: -32, dur: 0.55 },
        { db: -38, dur: 0.65 }
      ];
      
      for (const t of fallbackThresholds) {
        const testPoints = detectSilencePoints(channelData, sampleRate, t.db, t.dur);
        if (testPoints.length + 1 === expectedCount) {
          splitPoints = testPoints;
          slicesCount = expectedCount;
          fallbackLevel = 1;
          addLog(`[Batch ${batch.id}] 一级降级匹配成功 (阈值: ${t.db}dB, 持续: ${t.dur}s)!`, "success");
          break;
        }
      }
    }
    
    // 【强制拦截】：静音切割失败后，彻底放弃危险的字数估算（极易导致上下句切断尾音）
    if (slicesCount !== expectedCount) {
      addLog(`[Batch ${batch.id}] 静态边界切分失败。拦截错位风险，抛出异常交由 Level 3 逐句重生保底...`, "warn");
      throw new Error("LEVEL_3_FALLBACK_REQUIRED");
    }
    
    // 按时间点切割并封装 WAV
    const slices: { seq: number; blob: Blob; url: string; fallbackLevel?: number }[] = [];
    const fullTimeLine = [0, ...splitPoints, audioBuffer.duration];
    
    for (let k = 0; k < expectedCount; k++) {
      const startT = fullTimeLine[k];
      const endT = fullTimeLine[k+1];
      const slicedBlob = bufferToWav(audioBuffer, startT, endT);
      const slicedUrl = URL.createObjectURL(slicedBlob);
      const seq = batch.seqs[k];
      
      slices.push({ seq, blob: slicedBlob, url: slicedUrl, fallbackLevel });
      
      // 写入 IndexedDB 缓存
      db.set('audios', `${projectId}_${seq}`, slicedBlob).catch(err => console.error("IndexedDB audio save failed", err));

      // 更新到全局存储以备后续拼接
      setGlobalSequencedAudios(prev => {
        const map = new Map(prev);
        map.set(seq, slicedBlob);
        return map;
      });
      setGlobalSequencedUrls(prev => {
        const map = new Map(prev);
        map.set(seq, slicedUrl);
        return map;
      });
    }
    
    return slices;
  };

  const handleRunTtsPipeline = async (resume: boolean = false) => {
    setIsRunningTts(true);
    isRunningTtsRef.current = true;
    
    const keys = apiKeysStr.split('\n').map(k => k.trim()).filter(k => k);
    if (keys.length === 0) {
      addLog("未配置 API Key", "error");
      setIsRunningTts(false);
      return;
    }
    
    let activeBatches = resume 
      ? [...ttsBatches] 
      : ttsBatches.map(b => ({ 
          ...b, 
          status: 'pending', 
          audioBlob: undefined, 
          audioUrl: undefined, 
          slicedAudios: undefined, 
          error: undefined 
        }));
    
    if (!resume) {
      // 全新合成会整体丢弃旧批次资源，先释放其 ObjectURL 与播放器缓存，
      // 防止长篇反复重跑时数百个 blob URL 钉住音频内存持续增长
      Object.values(audioPlayersRef.current).forEach(p => p.pause());
      audioPlayersRef.current = {};
      ttsBatches.forEach(b => {
        if (b.audioUrl) URL.revokeObjectURL(b.audioUrl);
        if (b.slicedAudios) b.slicedAudios.forEach(s => URL.revokeObjectURL(s.url));
      });
      setTtsBatches(activeBatches);
    }
    
    addLog(`Layer 3: Dispatching ${activeBatches.length} TTS batches...`, "info");
    
    for (let i = 0; i < activeBatches.length; i++) {
      if (!isRunningTtsRef.current) break;
      const batch = activeBatches[i];
      if (batch.status === 'success') continue;
      
      setTtsBatches(prev => {
        const next = [...prev];
        const idx = next.findIndex(b => b.id === batch.id);
        if (idx !== -1) next[idx].status = 'generating';
        return next;
      });
      
      try {
        const emotionDesc = emotionDescriptions[batch.emotionClass] || "平静、自然";
        const formattedPrompt = `请用${emotionDesc}的情绪，朗读以下内容。
遇到 [pause=0.8] 标记时，请保持完全静音停顿 1 秒钟，绝对不要朗读或发出该标记的读音。
方括号括起来的英文内容是配音风格指导（例如 [shouting, furious]），请根据该指导调整朗读的语气和节奏，且绝对不要发出方括号内的任何读音。

朗读内容：
${batch.text}`;
        const wavBlob = await callGeminiTTSWithRetry(formattedPrompt, batch.voiceId, 3);
        const sliced = await sliceAndValidateBatch(wavBlob, batch);
        
        setTtsBatches(prev => {
          const next = [...prev];
          const idx = next.findIndex(b => b.id === batch.id);
          if (idx !== -1) {
            next[idx].status = 'success';
            next[idx].audioBlob = wavBlob;
            next[idx].audioUrl = URL.createObjectURL(wavBlob);
            next[idx].slicedAudios = sliced;
          }
          return next;
        });
        
        addLog(`✅ Batch [${batch.id}] 生成并完成切割（Level: ${sliced[0]?.fallbackLevel ?? 0}）。`, "success");
      } catch (err: any) {
        addLog(`❌ Batch [${batch.id}] 合成失败: ${err.message}`, "error");
        setTtsBatches(prev => {
          const next = [...prev];
          const idx = next.findIndex(b => b.id === batch.id);
          if (idx !== -1) {
            next[idx].status = 'failed';
            next[idx].error = err.message;
          }
          return next;
        });

        // 自动触发拆包单句合成，百分百解决串音与错位死角
        if (err.message === "LEVEL_3_FALLBACK_REQUIRED") {
           await handleRegenerateBatchSentenceBySentence(batch.id, true);
        }
      }
    }
    
    setIsRunningTts(false);
    isRunningTtsRef.current = false;
    
    // activeBatches 的元素引用与循环内 setState 的突变共享，
    // 用它判定完成状态；渲染时闭包里的 ttsBatches 是旧快照，永远判定不出"全部完成"
    if (activeBatches.every(b => b.status === 'success' || b.status === 'failed')) {
       setPipelinePhase('tts');
       setWorkspaceTab('master');
    }
  };

  // 单条对白单独重新合成（应急调节机制）
  const handleRegenerateSingleSentence = async (seq: number) => {
    const seg = globalSegments.find(s => s.seq === seq);
    if (!seg) return;
    
    setRegeneratingSeq(seq);
    addLog(`手动应急重生成对白 [seq:${seq}]: "${seg.text.substring(0, 15)}..."`, "info");
    
    try {
      const keys = apiKeysStr.split('\n').map(k => k.trim()).filter(k => k);
      if (keys.length === 0) throw new Error("无可用 API Key");
      
      const emotionDesc = emotionDescriptions[seg.emotion_class] || "平静、自然";
      const singleText = `请用${emotionDesc}的情绪，朗读以下台词。遇到省略号时，请保持完全静音停顿1秒钟，绝对不要读出标点符号。

台词内容：
……
${seg.text}`;
      const blob = await callGeminiTTSWithRetry(singleText, seg.voice, 3);
      const url = URL.createObjectURL(blob);
      
      // 写入 IndexedDB 缓存
      await db.set('audios', `${projectId}_${seq}`, blob);

      // 销毁旧播放器：缓存的 HTMLAudioElement 仍持有旧 blob URL，不清理会继续播放修改前的音频
      const oldPlayer = audioPlayersRef.current[seq];
      if (oldPlayer) {
        oldPlayer.pause();
        delete audioPlayersRef.current[seq];
      }

      setGlobalSequencedAudios(prev => new Map(prev).set(seq, blob));
      setGlobalSequencedUrls(prev => {
        const oldUrl = prev.get(seq);
        if (oldUrl) URL.revokeObjectURL(oldUrl);
        return new Map(prev).set(seq, url);
      });
      
      addLog(`✅ 对白 [seq:${seq}] 单句合成并更新完成。`, "success");
    } catch (err: any) {
      addLog(`❌ 对白 [seq:${seq}] 合成失败: ${err.message}`, "error");
    } finally {
      setRegeneratingSeq(null);
    }
  };

  // Layer 3.5: 三级降级 - 逐句单独合成整个批次 (Level 3 保底重合成)
  const handleRegenerateBatchSentenceBySentence = async (batchId: string, bypassConfirm: boolean = false) => {
    const batch = ttsBatches.find(b => b.id === batchId);
    if (!batch) return;
    
    if (!bypassConfirm && !window.confirm(`确定要启动 Level 3 保底重合成吗？系统将针对此批次内共 ${batch.seqs.length} 句对白分别调用 TTS 接口生成单独音频，这能100%保证切割完美，但会消耗较多 API 请求。`)) return;
    
    addLog(`启动 Level 3 逐句单独重合成批次 [${batchId}]...`, "info");
    
    setTtsBatches(prev => prev.map(b => b.id === batchId ? { ...b, status: 'generating' } : b));
    
    try {
      const slices = [];
      
      for (let i = 0; i < batch.seqs.length; i++) {
        const seq = batch.seqs[i];
        const seg = globalSegments.find(s => s.seq === seq);
        if (!seg) continue;
        
        addLog(`[Batch ${batchId}] 正在合成第 ${i+1}/${batch.seqs.length} 句 (seq:${seq})...`, "info");
        
        const emotionDesc = emotionDescriptions[seg.emotion_class] || "平静、自然";
        const singleText = `请用${emotionDesc}的情绪，朗读以下内容。绝对不要朗读或发出方括号内的任何读音。\n\n朗读内容：\n${seg.audio_tag} ${seg.text}`;
        
        const blob = await callGeminiTTSWithRetry(singleText, seg.voice, 3);
        const url = URL.createObjectURL(blob);
        
        // 保存至 IndexedDB
        await db.set('audios', `${projectId}_${seq}`, blob);
        
        // 更新 Map
        setGlobalSequencedAudios(prev => new Map(prev).set(seq, blob));
        setGlobalSequencedUrls(prev => new Map(prev).set(seq, url));
        
        slices.push({ seq, blob, url, fallbackLevel: 3 });
      }
      
      setTtsBatches(prev => prev.map(b => {
        if (b.id !== batchId) return b;
        // 被替换的旧批次资源不再被任何状态引用（逐句新 URL 已在上面写入），统一释放
        if (b.audioUrl) URL.revokeObjectURL(b.audioUrl);
        if (b.slicedAudios) b.slicedAudios.forEach(s => URL.revokeObjectURL(s.url));
        return { 
          ...b, 
          status: 'success',
          slicedAudios: slices,
          audioBlob: undefined,
          audioUrl: undefined
        };
      }));
      
      addLog(`✅ Batch [${batchId}] Level 3 逐句单独重合成成功！共合成 ${slices.length} 个片段。`, "success");
    } catch (err) {
      addLog(`❌ Batch [${batchId}] Level 3 重合成失败: ${err.message}`, "error");
      setTtsBatches(prev => prev.map(b => b.id === batchId ? { ...b, status: 'failed', error: err.message } : b));
    }
  };

  // 清空工程重置机制
  const handleClearProject = async () => {
    if (!window.confirm("确定要清空当前工程的所有数据和生成的音频缓存吗？此操作无法撤销。")) return;
    
    setIsDbLoading(true);
    try {
      await db.clear('state');
      await db.clear('audios');
      
      setRawText('');
      setCastJsonStr('');
      setCastData(null);
      setChunks([]);
      setTtsBatches([]);
      setGlobalSegments([]);
      setGlobalSequencedAudios(new Map());
      
      releaseAllAudioResources();
      setGlobalSequencedUrls(new Map());
      
      setMasterAudioBlob(null);
      setMasterAudioUrl(null);
      setPipelinePhase('setup');
      setSelectedChunkId(null);
      setSelectedBatchId(null);
      setCurrentPage(1);
      
      setNovelFileName('');
      
      addLog("工程已成功清空！", "success");
    } catch (err) {
      addLog("清空工程失败：" + err.message, "error");
    } finally {
      setIsDbLoading(false);
    }
  };

  // ==========================================
  // 12. Layer 4: 全局 seq 时间轴有序拼接
  // ==========================================
  const handleStitchMasterAudio = async () => {
    if (globalSegments.length === 0) {
      addLog("当前无可用剧本，请先执行导入和打标。", "error");
      return;
    }
    
    setPipelinePhase('stitching');
    addLog(`Layer 4: Starting ordered audio stitching for ${globalSegments.length} clips...`, "info");
    
    try {
      const missingSeqs: number[] = [];
      for (let seq = 1; seq <= globalSegments.length; seq++) {
        if (!globalSequencedAudios.has(seq)) {
          missingSeqs.push(seq);
        }
      }
      
      if (missingSeqs.length > 0) {
        throw new Error(`缺少以下 seq 的片段音频: ${missingSeqs.join(', ')}。请先将其合成。`);
      }
      
      let totalPcmLength = 0;
      for (let seq = 1; seq <= globalSegments.length; seq++) {
        totalPcmLength += globalSequencedAudios.get(seq)!.size - 44;
      }

      // 启用文件系统写流替代前端内存叠加，彻底消灭 OOM (内存溢出) 崩溃
      // @ts-ignore
      const fileHandle = await window.showSaveFilePicker({
        suggestedName: `${novelFileName || castData?.novel_name || 'audiobook'}_母带合并.wav`,
        types: [{ description: 'WAV Audio File', accept: { 'audio/wav': ['.wav'] } }],
      });
      
      // @ts-ignore
      const writable = await fileHandle.createWritable();
      addLog("🚀 已建立本地硬盘 IO 直连，正在边合并边静默落盘...", "info");
      
      const dummyPcm = new Uint8Array(0);
      const dummyBlob = pcmToWav(dummyPcm, 24000);
      const headerBuffer = await dummyBlob.arrayBuffer();
      const view = new DataView(headerBuffer);
      view.setUint32(4, 36 + totalPcmLength, true); 
      view.setUint32(40, totalPcmLength, true); 
      await writable.write(headerBuffer);

      for (let seq = 1; seq <= globalSegments.length; seq++) {
        const blob = globalSequencedAudios.get(seq)!;
        const ab = await blob.arrayBuffer();
        const pcmBytes = new Uint8Array(ab, 44);
        await writable.write(pcmBytes);
        
        if (seq % 500 === 0) addLog(`🔄 硬盘持续写入中... 进度: ${seq} / ${globalSegments.length} 句`, "info");
      }
      
      await writable.close();
      
      setMasterAudioBlob(new Blob(["(已导出到本地硬盘)"])); 
      setMasterAudioUrl(""); // UI 按钮隐藏
      addLog(`🎉 无损母带拼合完毕并直接落盘！文件大小: ${(totalPcmLength / 1024 / 1024).toFixed(2)} MB. 请前往您保存的目录查看。`, "success");
    } catch (err: any) {
      addLog(`❌ 拼接失败: ${err.message}`, "error");
    } finally {
      setPipelinePhase('tts');
    }
  };

  // 辅助下载功能：导出本地 FFmpeg 拼接描述文本和 bat 脚本
  const handleExportFFmpegFiles = () => {
    let listContent = "";
    for (let seq = 1; seq <= globalSegments.length; seq++) {
      listContent += `file 'seq_${seq}.wav'\r\n`;
    }
    
    // 导出 concat_list.txt
    const txtBlob = new Blob([listContent], { type: 'text/plain' });
    const txtUrl = URL.createObjectURL(txtBlob);
    const a1 = document.createElement('a');
    a1.href = txtUrl;
    const baseName = novelFileName || castData?.novel_name || 'audiobook';
    a1.download = `${baseName}_导出_concat_list.txt`;
    a1.click();
    
    // 导出 concat.bat 批处理文件
    const batContent = `@echo off\r\necho Concatenating waves using FFmpeg...\r\nffmpeg -f concat -safe 0 -i ${baseName}_导出_concat_list.txt -c copy ${baseName}_导出.wav\r\necho Concatenation Completed!\r\npause`;
    const batBlob = new Blob([batContent], { type: 'application/octet-stream' });
    const batUrl = URL.createObjectURL(batBlob);
    const a2 = document.createElement('a');
    a2.href = batUrl;
    a2.download = `${baseName}_导出_concat.bat`;
    a2.click();
    
    addLog("FFmpeg 脚本文件 concat_list.txt & concat.bat 导出成功。", "success");
  };

  // 导出 TTS 数据集功能（给 TTS 之前的数据集合打包下载）
  const handleExportTTSDataset = () => {
    if (ttsBatches.length === 0) {
      addLog("当前无可用 TTS 批次，请先运行分块和打标。", "warn");
      return;
    }

    const exportData = {
      novel_name: castData?.novel_name || "audiobook",
      tagging_model: modelName,
      tts_model: ttsModelName,
      global_segments: globalSegments,
      tts_batches: ttsBatches.map(b => ({
        id: b.id,
        speaker: b.speaker,
        voiceId: b.voiceId,
        emotionClass: b.emotionClass,
        text: b.text,
        seqs: b.seqs
      }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const baseName = novelFileName || exportData.novel_name || 'audiobook';
    a.download = `${baseName}_tts_dataset.json`;
    a.click();
    URL.revokeObjectURL(url);
    addLog(`✅ TTS 前置数据集导出成功：${baseName}_tts_dataset.json`, "success");
  };

  const handlePlayToggle = (seq: number) => {
    const audioUrl = globalSequencedUrls.get(seq);
    if (!audioUrl) return;
    
    if (playingSeq === seq) {
      audioPlayersRef.current[seq]?.pause();
      setPlayingSeq(null);
    } else {
      // 停止正在播放的其它片段
      if (playingSeq !== null && audioPlayersRef.current[playingSeq]) {
        audioPlayersRef.current[playingSeq].pause();
      }
      
      if (!audioPlayersRef.current[seq]) {
        const audio = new Audio(audioUrl);
        audio.onended = () => setPlayingSeq(null);
        audioPlayersRef.current[seq] = audio;
      }
      audioPlayersRef.current[seq].play();
      setPlayingSeq(seq);
    }
  };

  const updateBatchStatus = (id: string, status: 'pending'|'generating'|'success'|'failed', err?: string) => {
    setTtsBatches(prev => prev.map(b => b.id === id ? { ...b, status, error: err } : b));
  };

  const progress = chunks.length > 0 ? (chunks.filter(c => c.status === 'success').length / chunks.length) * 100 : 0;
  const strokeDashoffset = 125 - (125 * progress) / 100;

  const ttsProgress = ttsBatches.length > 0 ? (ttsBatches.filter(b => b.status === 'success').length / ttsBatches.length) * 100 : 0;

  if (isDbLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-[#07070e] text-slate-200">
        <Loader2 className="w-10 h-10 text-indigo-500 animate-spin mb-4 text-indigo-400" />
        <h3 className="font-bold text-sm tracking-wider text-indigo-400">正在载入工程数据及音频缓存...</h3>
        <p className="text-xs text-slate-500 mt-2">请稍候，我们正在从本地数据库恢复您的进度</p>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-[#07070e] text-slate-200">
      
      {/* 🟢 左侧：工程导航与管线进度 (Pipeline Navigator) */}
      <div className="w-48 border-r border-slate-800/50 glass-panel flex flex-col z-10 shadow-2xl">
        <div className="p-3 border-b border-slate-800/50">
          <div className="flex flex-col gap-2 mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl gradient-accent flex items-center justify-center shadow-lg shadow-indigo-500/20">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <div className="overflow-hidden flex-1">
                <h1 className="font-bold text-lg tracking-tight text-white">AudioLab v2</h1>
                {novelFileName ? (
                  <p className="text-[10px] text-indigo-400 font-semibold truncate" title={novelFileName}>
                    📖 {novelFileName}
                  </p>
                ) : (
                  <p className="text-[10px] text-slate-400 font-mono tracking-widest uppercase">多音色有声书管线</p>
                )}
              </div>
            </div>
            <button 
              onClick={() => setIsProjectModalOpen(true)}
              className="mt-2 flex items-center justify-center gap-2 w-full py-1.5 px-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg border border-slate-700/50 text-xs font-semibold text-slate-300 transition-colors"
            >
              <FolderOpen className="w-3.5 h-3.5" />
              历史任务管理
            </button>
          </div>

          <div className="flex items-center gap-4 p-4 glass-card bg-slate-900/40">
            <div className="relative w-12 h-12">
              <svg className="w-12 h-12" viewBox="0 0 44 44">
                <circle className="text-slate-800" strokeWidth="3" stroke="currentColor" fill="transparent" r="20" cx="22" cy="22" />
                <circle className="text-indigo-500 progress-ring-circle" strokeWidth="3" strokeDasharray="125" strokeDashoffset={strokeDashoffset} strokeLinecap="round" stroke="currentColor" fill="transparent" r="20" cx="22" cy="22" />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center font-mono text-[10px] font-bold text-indigo-400">
                {Math.round(progress)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-400 font-semibold">AI 剧本打标进度</div>
              <div className="font-mono text-sm font-bold text-indigo-300">{chunks.filter(c => c.status === 'success').length} / {chunks.length} 分块</div>
            </div>
          </div>
        </div>

        {/* Chunks Navigation */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2 custom-scrollbar">
          <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest px-2 mb-2">原著小说分块目录 (Layer 1)</div>
          {chunks.length === 0 ? (
             <div className="text-center text-slate-600 mt-10 text-xs px-4">
               等待导入原著文本...
             </div>
          ) : chunks.map((chunk, idx) => (
            <div 
              key={chunk.id} 
              onClick={() => {
                setSelectedChunkId(chunk.id);
                setWorkspaceTab('canvas');
              }}
              className={cn(
                "p-2 rounded-lg border cursor-pointer transition-all duration-200 group flex items-center justify-between",
                selectedChunkId === chunk.id && workspaceTab === 'canvas' ? "bg-indigo-500/10 border-indigo-500/30" : "hover:bg-slate-800/40 border-transparent"
              )}
            > 
              <div className="flex items-center gap-2">
                <input 
                  type="checkbox" 
                  className="w-3 h-3 rounded bg-slate-900 border-slate-700 accent-indigo-500 cursor-pointer shrink-0"
                  checked={selectedChunkIds.has(chunk.id)}
                  onChange={(e) => {
                    e.stopPropagation();
                    setSelectedChunkIds(prev => {
                      const next = new Set(prev);
                      if (next.has(chunk.id)) next.delete(chunk.id); else next.add(chunk.id);
                      return next;
                    });
                  }}
                />
                <div className={cn(
                  "w-1.5 h-1.5 rounded-full shrink-0",
                  chunk.status === 'success' ? "bg-green-400" :
                  chunk.status === 'processing' ? "bg-indigo-400 animate-pulse" :
                  chunk.status === 'rescued' ? "bg-amber-400" :
                  chunk.status === 'failed' ? "bg-red-400" : "bg-slate-600"
                )} />
                <span className={cn("text-[10px] font-mono truncate w-14", selectedChunkId === chunk.id && workspaceTab === 'canvas' ? "text-indigo-300 font-bold" : "text-slate-400")}>
                  #{idx + 1}
                </span>
              </div>
              <span className="text-[9px] text-slate-500 font-mono shrink-0 ml-auto">{chunk.originalText.length}字</span>
            </div>
          ))}
        </div>
        
        {/* Run Controls */}
        <div className="p-4 border-t border-slate-800/50 bg-[#0f0f1e] space-y-2 shrink-0">
          <div className="flex items-center gap-2 text-[10px] text-slate-400 bg-slate-900/50 p-1.5 rounded-lg border border-slate-800/50">
            <span className="font-semibold shrink-0">分块字数:</span>
            <input 
              type="number" 
              value={chunkSize}
              onChange={(e) => setChunkSize(Number(e.target.value))}
              placeholder="默认: 1000"
              className="bg-transparent border border-slate-700 rounded px-1.5 py-0.5 w-full text-center text-indigo-300 font-mono outline-none [&::-webkit-inner-spin-button]:appearance-none"
            />
            <button 
              onClick={() => {
                if (chunks.some(c => c.status === 'success') && !window.confirm("警告：重新分块将丢失所有已打标进度！确定继续？")) return;
                handleParseAndChunk();
              }}
              disabled={!rawText || !castJsonStr}
              className="bg-indigo-600 hover:bg-indigo-500 text-white px-2 py-0.5 rounded shrink-0 transition-colors disabled:opacity-40 cursor-pointer"
            >
              应用
            </button>
          </div>

          {chunks.length === 0 ? (
            <button 
              onClick={handleParseAndChunk} 
              disabled={!rawText || !castJsonStr}
              className="w-full flex justify-center items-center gap-2 py-2 rounded-lg font-bold transition-all text-xs cursor-pointer btn-primary shadow-lg disabled:opacity-40 disabled:pointer-events-none"
            >
              <Upload className="w-4 h-4 text-indigo-400" /> 1. 智能分块文本 (Smart Chunking)
            </button>
          ) : (() => {
            const hasFinishedSomeChunks = chunks.some(c => c.status === 'success');
            const hasUnfinishedChunks = chunks.some(c => c.status !== 'success');
            return (
              <div className="space-y-2 text-xs">
                {hasFinishedSomeChunks && hasUnfinishedChunks ? (
                  <div className="flex flex-col gap-2">
                    <button 
                      onClick={() => isRunning ? (isRunningRef.current = false) : processAllChunks(true)}
                      className={cn(
                        "w-full flex justify-center items-center gap-2 py-2 rounded-lg font-bold transition-all cursor-pointer shadow-lg",
                        isRunning ? "bg-red-500/20 text-red-400 border border-red-500/30" : "bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-500/10 hover:shadow-emerald-500/20"
                      )}>
                      {isRunning ? <><Pause className="w-3 h-3" /> 停止</> : <><Play className="w-3 h-3" /> 继续任务</>}
                    </button>
                    <button 
                      onClick={() => {
                        if (window.confirm("确定要放弃当前的进度，重新开始全部打标吗？")) {
                          processAllChunks(false);
                        }
                      }}
                      disabled={isRunning}
                      className="w-full flex justify-center items-center gap-2 py-1.5 rounded-lg border border-slate-700 bg-slate-800/40 text-slate-350 hover:bg-slate-800 transition-colors disabled:opacity-40 cursor-pointer"
                    >
                      <RefreshCw className="w-3 h-3 text-slate-400" /> 重新全部打标
                    </button>
                  </div>
                ) : (
                  <button 
                    onClick={() => isRunning ? (isRunningRef.current = false) : processAllChunks(false)}
                    className={cn(
                      "w-full flex justify-center items-center gap-2 py-2 rounded-lg font-bold transition-all cursor-pointer",
                      isRunning ? "bg-red-500/20 text-red-400 border border-red-500/30" : "btn-primary"
                    )}>
                    {isRunning ? <><Pause className="w-3 h-3" /> 停止</> : <><Play className="w-3 h-3" /> 批量打标</>}
                  </button>
                )}
                  
                <div className="flex gap-2">
                  <button onClick={handleSelectFailed} className="flex-1 border border-slate-700 bg-slate-800 hover:bg-slate-700 py-1.5 rounded-lg text-[10px] cursor-pointer">
                    全选失败
                  </button>
                  <button onClick={handleRetrySelectedChunks} disabled={selectedChunkIds.size === 0} className="flex-1 bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 rounded-lg py-1.5 text-[10px] disabled:opacity-30 cursor-pointer">
                    补扫({selectedChunkIds.size})
                  </button>
                </div>
                  
                <button 
                  onClick={() => buildTTSBatches(chunks.filter(c => c.status === 'success' || c.status === 'rescued'))}
                  className="w-full border border-indigo-500/30 bg-indigo-900/20 text-indigo-300 hover:bg-indigo-900/40 py-2 rounded-lg text-[10px] font-bold cursor-pointer"
                >
                  重新聚合批次 (L2.5)
                </button>
              </div>
            );
          })()}
        </div>
      </div>

      {/* 🟢 中间：沉浸式多功能剧本画布 (Multifunctional Workspace Canvas) */}
      <div className="flex-1 flex flex-col relative bg-[#0a0a14] shadow-inner overflow-hidden">
        
        {/* Workspace Mode Tabs */}
        <div className="absolute top-0 inset-x-0 h-16 glass-panel border-b border-slate-800/50 z-20 flex items-center justify-between px-6">
          <div className="flex items-center gap-2 overflow-hidden mr-4">
            <Music className="w-4 h-4 text-indigo-400 shrink-0" />
            <h2 className="font-semibold text-slate-200 shrink-0 mr-4">工作区</h2>
            {novelFileName && (
              <span className="text-xs text-indigo-400 font-medium px-2 py-0.5 rounded bg-slate-900 border border-slate-800 truncate max-w-[150px] mr-4" title={novelFileName}>
                📖 {novelFileName}
              </span>
            )}
            <div className="flex bg-slate-900/80 p-0.5 rounded-lg border border-slate-850 shrink-0">
              <button 
                onClick={() => setWorkspaceTab('canvas')}
                className={cn("px-3 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer", workspaceTab === 'canvas' ? "bg-indigo-600 text-white shadow-md" : "text-slate-400 hover:text-slate-200")}
              >
                <Layers className="w-3.5 h-3.5" /> 1. 剧本打标切片
              </button>
              <button 
                onClick={() => setWorkspaceTab('batches')}
                className={cn("px-3 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer", workspaceTab === 'batches' ? "bg-indigo-600 text-white shadow-md" : "text-slate-400 hover:text-slate-200")}
              >
                <Users className="w-3.5 h-3.5" /> 2. TTS 语音合成队列
              </button>
              <button 
                onClick={() => setWorkspaceTab('master')}
                className={cn("px-3 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer", workspaceTab === 'master' ? "bg-indigo-600 text-white shadow-md" : "text-slate-400 hover:text-slate-200")}
              >
                <Music className="w-3.5 h-3.5" /> 3. 整书拼接与下载
              </button>
            </div>
          </div>
          
          <div className="flex items-center gap-2 shrink-0">
            {/* Tagging Model Selector & Custom Input */}
            <div className="flex flex-col justify-center bg-slate-900/60 border border-slate-800 rounded px-2 py-1 text-[10px] w-36">
              <div className="flex items-center justify-between gap-1 border-b border-slate-800/40 pb-0.5">
                <span className="text-slate-500 font-mono shrink-0">打标:</span>
                <select 
                  value={['gemini-3.1-flash-lite', 'gemini-3.5-flash', 'gemini-3-flash'].includes(modelName) ? modelName : 'custom'}
                  onChange={e => {
                    if (e.target.value !== 'custom') {
                      setModelName(e.target.value);
                    }
                  }}
                  className="bg-transparent text-slate-300 outline-none font-mono cursor-pointer text-[10px] text-right w-full"
                >
                  <option value="gemini-3.1-flash-lite">3.1-flash-lite</option>
                  <option value="gemini-3.5-flash">3.5-flash</option>
                  <option value="gemini-3-flash">3-flash</option>
                  <option value="custom">自定义...</option>
                </select>
              </div>
              <input 
                type="text" 
                value={modelName}
                onChange={e => setModelName(e.target.value)}
                className="bg-transparent text-indigo-300 font-mono outline-none text-[9px] mt-0.5 text-center"
                placeholder="输入模型..."
              />
            </div>

            {/* TTS Model Selector & Custom Input */}
            <div className="flex flex-col justify-center bg-slate-900/60 border border-slate-800 rounded px-2 py-1 text-[10px] w-36">
              <div className="flex items-center justify-between gap-1 border-b border-slate-800/40 pb-0.5">
                <span className="text-slate-500 font-mono shrink-0">合成:</span>
                <select 
                  value={['gemini-3.1-flash-tts-preview', 'gemini-3.1-flash-lite', 'gemini-3.5-flash', 'gemini-3-flash'].includes(ttsModelName) ? ttsModelName : 'custom'}
                  onChange={e => {
                    if (e.target.value !== 'custom') {
                      setTtsModelName(e.target.value);
                    }
                  }}
                  className="bg-transparent text-slate-300 outline-none font-mono cursor-pointer text-[10px] text-right w-full"
                >
                  <option value="gemini-3.1-flash-tts-preview">3.1-flash-tts</option>
                  <option value="gemini-3.1-flash-lite">3.1-flash-lite</option>
                  <option value="gemini-3.5-flash">3.5-flash</option>
                  <option value="gemini-3-flash">3-flash</option>
                  <option value="custom">自定义...</option>
                </select>
              </div>
              <input 
                type="text" 
                value={ttsModelName}
                onChange={e => setTtsModelName(e.target.value)}
                className="bg-transparent text-indigo-300 font-mono outline-none text-[9px] mt-0.5 text-center"
                placeholder="输入模型..."
              />
            </div>

            {/* Toggle Right Panel Button */}
            <button 
              onClick={() => setIsRightPanelCollapsed(!isRightPanelCollapsed)}
              className="p-1.5 ml-1 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
              title={isRightPanelCollapsed ? "显示配置与日志" : "隐藏配置与日志"}
            >
              {isRightPanelCollapsed ? <PanelRightOpen className="w-4 h-4" /> : <PanelRightClose className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Tab 1: Tagged Chunks Canvas */}
        {workspaceTab === 'canvas' && (
          <div className="flex-1 pt-20 pb-4 px-6 flex flex-col min-h-0 overflow-hidden">
            {chunks.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center opacity-40">
                <div className="w-20 h-20 mb-6 rounded-3xl gradient-accent opacity-20 flex items-center justify-center">
                  <FileText className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-lg font-bold mb-2">需要初始化原著文本</h3>
                <p className="text-xs text-slate-400 max-w-sm text-center">请直接点击左下角「1. 智能分块文本」启动，或在右侧「Setup」配置面板修改小说和选角字典。</p>
              </div>
            ) : (() => {
              const activeIdx = chunks.findIndex(c => c.id === selectedChunkId);
              const activeChunk = activeIdx !== -1 ? chunks[activeIdx] : chunks[0];
              const displayIdx = activeIdx !== -1 ? activeIdx : 0;
              
              return (
                <div className="flex-1 flex flex-col min-h-0 space-y-4">
                  {/* Chunk Control Bar */}
                  <div className="flex items-center justify-between bg-slate-900/60 border border-slate-800/80 px-4 py-2.5 rounded-xl shrink-0">
                    <div className="flex items-center gap-3">
                      <span className="font-bold text-xs text-slate-200">当前查看：小说分块 #{displayIdx + 1}</span>
                      <span className={cn(
                        "badge text-[9px] uppercase px-2 py-0.5 rounded",
                        activeChunk.status === 'success' ? 'badge-success' :
                        activeChunk.status === 'processing' ? 'badge-processing' :
                        activeChunk.status === 'rescued' ? 'badge-rescued' :
                        activeChunk.status === 'failed' ? 'badge-failed' : 'badge-pending'
                      )}>
                        {activeChunk.status === 'success' ? '打标成功' :
                         activeChunk.status === 'processing' ? '正在打标' :
                         activeChunk.status === 'rescued' ? '避灾分割' :
                         activeChunk.status === 'failed' ? '打标失败' : '等待打标'}
                      </span>
                      <span className="text-[10px] font-mono text-slate-500">{activeChunk.originalText.length} 字</span>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <button 
                        onClick={() => {
                          if (displayIdx > 0) {
                            setSelectedChunkId(chunks[displayIdx - 1].id);
                          }
                        }}
                        disabled={displayIdx === 0}
                        className="px-2.5 py-1.5 rounded-lg bg-slate-850 hover:bg-slate-800 text-xs font-semibold text-slate-300 disabled:opacity-30 disabled:pointer-events-none transition-colors cursor-pointer"
                      >
                        上一分块
                      </button>
                      <button 
                        onClick={() => {
                          if (displayIdx < chunks.length - 1) {
                            setSelectedChunkId(chunks[displayIdx + 1].id);
                          }
                        }}
                        disabled={displayIdx === chunks.length - 1}
                        className="px-2.5 py-1.5 rounded-lg bg-slate-850 hover:bg-slate-800 text-xs font-semibold text-slate-300 disabled:opacity-30 disabled:pointer-events-none transition-colors cursor-pointer"
                      >
                        下一分块
                      </button>
                    </div>
                  </div>

                  {/* Side-by-side split screen */}
                  <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-0">
                    {/* Left 1/2: Original Text */}
                    <div className="glass-card p-5 flex flex-col min-h-0 bg-slate-950/20">
                      <div className="flex justify-between items-center pb-3 border-b border-slate-800/40 mb-3 shrink-0">
                        <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">原著分块内容</span>
                      </div>
                      <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar text-slate-300 text-[13px] font-serif leading-loose whitespace-pre-wrap select-text">
                        {activeChunk.originalText}
                      </div>
                    </div>

                    {/* Right 1/2: Tagged Script */}
                    <div className="glass-card p-5 flex flex-col min-h-0 bg-slate-950/20">
                      <div className="flex justify-between items-center pb-3 border-b border-slate-800/40 mb-3 shrink-0">
                        <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">AI 剧本打标结果 (L2)</span>
                      </div>
                      
                      <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar space-y-4">
                        {activeChunk.status === 'success' && activeChunk.segments ? (
                          activeChunk.segments.map((seg, sIdx) => {
                            const isNarrator = seg.char === '旁白' || seg.char === 'Narrator';
                            const colorHex = getSpeakerColor(seg.char, castData?.cast || []);
                            
                            if (isNarrator) {
                              return (
                                <div key={sIdx} className="w-full pl-3 relative border-l-2 border-slate-700/50 py-1">
                                  <div className="text-[10px] text-slate-500 font-bold mb-1">旁白</div>
                                  <div className="text-slate-350 text-xs leading-relaxed">{seg.text}</div>
                                </div>
                              );
                            }

                            return (
                              <div key={sIdx} className="flex flex-col items-start w-full relative pl-3 border-l-2 py-1" style={{ borderLeftColor: colorHex }}>
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="font-bold text-[11px]" style={{ color: colorHex }}>{seg.char}</span>
                                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-indigo-300/80 font-mono">{seg.voice}</span>
                                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-500 italic font-mono">{seg.audio_tag}</span>
                                </div>
                                <div className="text-slate-200 text-xs leading-relaxed font-medium bg-slate-900/30 px-3 py-2 rounded-lg border border-slate-800/40 w-full">
                                  {seg.text}
                                </div>
                              </div>
                            );
                          })
                        ) : (
                          <div className="text-slate-500 whitespace-pre-wrap font-serif leading-loose text-xs opacity-75 max-h-40 overflow-hidden relative">
                            {activeChunk.status === 'processing' ? (
                              <div className="space-y-2 py-2">
                                <div className="h-3 bg-slate-850 rounded w-full animate-pulse"></div>
                                <div className="h-3 bg-slate-850 rounded w-5/6 animate-pulse"></div>
                              </div>
                            ) : activeChunk.status === 'failed' ? (
                              <div className="text-center py-10 text-xs text-red-500/80 font-bold">❌ AI 打标失败，请点击左下角“全选失败”及“补扫”重试。</div>
                            ) : activeChunk.status === 'rescued' ? (
                              <div className="text-center py-10 text-xs text-amber-500/80 font-bold">⚠️ 该分块已被自救切片，等待下一次管线调度。</div>
                            ) : (
                              <div className="text-center py-10 text-xs text-slate-600">等待 AI 打标任务启动...</div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })()}
          </div>
        )}

        {/* Tab 2: TTS Batches Queue Panel */}
        {workspaceTab === 'batches' && (
          <div className="flex-1 pt-20 pb-4 px-6 flex flex-col min-h-0 overflow-hidden">
            <div className="flex flex-col min-h-0 h-full space-y-4">
              
              {/* Batches Header Status Card */}
              <div className="glass-card p-5 flex flex-col md:flex-row items-center justify-between gap-6 bg-slate-900/40 shrink-0">
                <div className="flex items-center gap-4">
                  <div className="relative w-14 h-14 shrink-0">
                    <svg className="w-14 h-14" viewBox="0 0 44 44">
                      <circle className="text-slate-800" strokeWidth="4" stroke="currentColor" fill="transparent" r="18" cx="22" cy="22" />
                      <circle className="text-indigo-500 progress-ring-circle" strokeWidth="4" strokeDasharray="113" strokeDashoffset={113 - (113 * ttsProgress) / 100} strokeLinecap="round" stroke="currentColor" fill="transparent" r="18" cx="22" cy="22" />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center font-mono text-[10px] font-bold text-indigo-400">
                      {Math.round(ttsProgress)}%
                    </div>
                  </div>
                  <div>
                    <h3 className="font-bold text-xs text-white">TTS 语音合成管道</h3>
                    <p className="text-[11px] text-slate-400">已按角色与情感聚合音频批次。双重兜底静音锚点物理切分已生效。</p>
                    <div className="text-[10px] text-indigo-300 font-semibold mt-0.5">已成功：{ttsBatches.filter(b => b.status === 'success').length} / {ttsBatches.length} 批次</div>
                  </div>
                </div>
                
                <div className="flex flex-wrap items-center gap-3">
                  <button 
                    onClick={handleExportTTSDataset}
                    disabled={ttsBatches.length === 0}
                    className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-bold transition-all bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-750 cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5 text-indigo-400" /> 导出数据集 JSON
                  </button>
                  {(() => {
                    const hasFinishedSomeBatches = ttsBatches.some(b => b.status === 'success');
                    const hasUnfinishedBatches = ttsBatches.some(b => b.status !== 'success');
                    
                    if (hasFinishedSomeBatches && hasUnfinishedBatches) {
                      return (
                        <div className="flex items-center gap-2">
                          <button 
                            onClick={() => {
                              if (window.confirm("确定要重置当前所有的合成音频，重新开始合成吗？")) {
                                handleRunTtsPipeline(false);
                              }
                            }}
                            disabled={isRunningTts || ttsBatches.length === 0}
                            className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-bold transition-all bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-750 cursor-pointer"
                          >
                            <RefreshCw className="w-3.5 h-3.5 text-slate-400" /> 全部重新合成
                          </button>
                          
                          <button 
                            onClick={() => isRunningTts ? (isRunningTtsRef.current = false) : handleRunTtsPipeline(true)}
                            disabled={ttsBatches.length === 0}
                            className={cn(
                              "flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold transition-all shadow-lg cursor-pointer",
                              isRunningTts ? "bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30" : 
                              "bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-500/10 hover:shadow-emerald-500/20"
                            )}
                          >
                            {isRunningTts ? <><Pause className="w-3.5 h-3.5" /> 停止合成</> : <><Play className="w-3.5 h-3.5" /> 继续合成任务</>}
                          </button>
                        </div>
                      );
                    }
                    
                    return (
                      <button 
                        onClick={() => isRunningTts ? (isRunningTtsRef.current = false) : handleRunTtsPipeline(false)}
                        disabled={ttsBatches.length === 0}
                        className={cn(
                          "flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold transition-all shadow-lg cursor-pointer",
                          isRunningTts ? "bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30" : 
                          "btn-primary shadow-indigo-500/20 hover:shadow-indigo-500/40"
                        )}
                      >
                        {isRunningTts ? <><Pause className="w-3.5 h-3.5" /> 停止合成</> : <><Play className="w-3.5 h-3.5" /> 开始批量语音合成 (L3)</>}
                      </button>
                    );
                  })()}
                </div>
              </div>

              {/* Side-by-side split screen */}
              {ttsBatches.length === 0 ? (
                <div className="flex-1 flex flex-col items-center justify-center opacity-45">
                  <div className="text-center p-12 text-slate-500 text-xs">
                    未加载合成批次。请先在左侧侧边栏启动“AI 剧本打标 (L2)”。
                  </div>
                </div>
              ) : (() => {
                const activeBatchIdx = ttsBatches.findIndex(b => b.id === selectedBatchId);
                const activeBatch = activeBatchIdx !== -1 ? ttsBatches[activeBatchIdx] : ttsBatches[0];
                
                return (
                  <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-0">
                    {/* Left half: Batches List */}
                    <div className="flex flex-col min-h-0 space-y-2 overflow-y-auto pr-2 custom-scrollbar">
                      <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest px-1 shrink-0">角色与情感批次队列</div>
                      {ttsBatches.map((batch, idx) => {
                        const speakerColor = getSpeakerColor(batch.speaker, castData?.cast || []);
                        const isActive = selectedBatchId === batch.id || (!selectedBatchId && idx === 0);
                        
                        return (
                          <div 
                            key={batch.id} 
                            onClick={() => setSelectedBatchId(batch.id)}
                            className={cn(
                              "glass-card p-3.5 border-l-4 transition-all cursor-pointer",
                              isActive ? "border-l-indigo-500 bg-indigo-950/10 shadow-md border-opacity-100" : "border-l-slate-750 hover:bg-slate-900/20 border-opacity-40"
                            )}
                            style={{ borderLeftColor: speakerColor }}
                          >
                            <div className="flex justify-between items-center mb-1">
                              <div className="flex items-center gap-2">
                                <span className="font-bold text-xs" style={{ color: speakerColor }}>{batch.speaker}</span>
                                <span className="text-[10px] text-slate-400 font-mono">({batch.emotionClass})</span>
                              </div>
                              <span className={cn(
                                "badge text-[8px] uppercase",
                                batch.status === 'success' ? 'badge-success' :
                                batch.status === 'generating' ? 'badge-processing animate-pulse' :
                                batch.status === 'failed' ? 'badge-failed' : 'badge-pending'
                              )}>
                                {batch.status === 'success' ? '成功' :
                                 batch.status === 'generating' ? '合成中' :
                                 batch.status === 'failed' ? '失败' : '排队中'}
                              </span>
                            </div>
                            <div className="flex justify-between items-center text-[10px] text-slate-500 font-mono">
                              <span>音色: {batch.voiceId}</span>
                              <span>{batch.seqs.length} 句 | {batch.text.replace(/\[pause=0.8\]/g, '').replace(/\[.*?\]/g, '').length} 字</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Right half: Selected Batch Details */}
                    <div className="glass-card p-5 flex flex-col min-h-0 bg-slate-950/20">
                      {activeBatch ? (
                        <div className="flex flex-col min-h-0 h-full space-y-4">
                          <div className="flex justify-between items-start pb-3 border-b border-slate-800/40 shrink-0">
                            <div>
                              <div className="flex items-center gap-2">
                                <h4 className="font-bold text-sm text-white">{activeBatch.speaker}</h4>
                                <span className="text-xs px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-indigo-300 font-mono">{activeBatch.emotionClass}</span>
                                <span className="text-[10px] text-slate-500 font-mono">{activeBatch.id}</span>
                              </div>
                              <div className="text-[11px] text-slate-400 mt-1">
                                分配音色: <span className="text-indigo-400 font-mono">{activeBatch.voiceId}</span> | 包含对白: {activeBatch.seqs.length} 句
                              </div>
                            </div>
                            
                            <div className="flex items-center gap-2">
                              {activeBatch.status === 'success' && activeBatch.audioUrl && (
                                <button 
                                  onClick={() => {
                                    const audio = new Audio(activeBatch.audioUrl);
                                    audio.play();
                                  }}
                                  className="p-1.5 hover:bg-slate-800 rounded text-indigo-400 cursor-pointer"
                                  title="播放整段合成音频"
                                >
                                  <Volume2 className="w-4 h-4" />
                                </button>
                              )}
                            </div>
                          </div>

                          {/* Fallback warnings */}
                          {activeBatch.status === 'success' && activeBatch.slicedAudios && activeBatch.slicedAudios[0]?.fallbackLevel === 2 && (
                            <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl flex items-start gap-2.5 text-xs text-amber-300 shrink-0">
                              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-amber-400" />
                              <div className="flex-1">
                                <div className="font-bold">⚠️ 检测到二级降级（时间戳估算切割）</div>
                                <p className="text-[10px] text-amber-400/80 mt-0.5 leading-relaxed">
                                  静音阈值匹配失败。系统已使用文本字数比例估算切分点，这可能会导致句尾或句首有些许杂音。建议点击下方按钮进行“逐句重新合成”以确保100%完美契合。
                                </p>
                                <button
                                  onClick={() => handleRegenerateBatchSentenceBySentence(activeBatch.id)}
                                  className="mt-2 px-2.5 py-1 bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 rounded font-bold text-[10px] transition-colors cursor-pointer"
                                >
                                  一键逐句单独重合成 (Level 3)
                                </button>
                              </div>
                            </div>
                          )}

                          {/* Level 3 manual action if not yet Level 3 */}
                          {activeBatch.status === 'success' && (!activeBatch.slicedAudios || activeBatch.slicedAudios[0]?.fallbackLevel !== 3) && (
                            <div className="flex justify-end shrink-0">
                              <button
                                onClick={() => handleRegenerateBatchSentenceBySentence(activeBatch.id)}
                                className="px-3 py-1.5 bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 border border-indigo-500/20 rounded-lg text-xs font-bold transition-all cursor-pointer flex items-center gap-1.5"
                              >
                                <RefreshCw className="w-3.5 h-3.5" /> 一键逐句单独重合成 (Level 3 保底)
                              </button>
                            </div>
                          )}

                          {/* Sliced Sentences List */}
                          <div className="flex-1 overflow-y-auto pr-1 custom-scrollbar space-y-3">
                            {activeBatch.seqs.map((seq) => {
                              const seg = globalSegments.find(s => s.seq === seq);
                              const hasSliced = activeBatch.slicedAudios?.some(a => a.seq === seq) || globalSequencedAudios.has(seq);
                              const slicedUrl = activeBatch.slicedAudios?.find(a => a.seq === seq)?.url || globalSequencedUrls.get(seq);
                              
                              return (
                                <div key={seq} className="p-3 bg-slate-900/40 border border-slate-850 rounded-xl flex items-center justify-between gap-3 text-xs">
                                  <div className="flex items-center gap-2.5 overflow-hidden">
                                    <span className="font-mono text-[10px] text-slate-500 font-bold shrink-0">#{seq}</span>
                                    <div className="flex flex-col gap-0.5 overflow-hidden">
                                      <span className="text-slate-200 line-clamp-2 leading-relaxed">{seg ? seg.text : ""}</span>
                                      {seg?.audio_tag && (
                                        <span className="text-[9px] text-slate-500 font-mono">{seg.audio_tag}</span>
                                      )}
                                    </div>
                                  </div>
                                  
                                  <div className="flex items-center gap-1.5 shrink-0">
                                    {hasSliced && slicedUrl ? (
                                      <button 
                                        onClick={() => handlePlayToggle(seq)}
                                        className="p-1.5 hover:bg-indigo-500/10 rounded text-indigo-400 cursor-pointer"
                                      >
                                        {playingSeq === seq ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                                      </button>
                                    ) : (
                                      <span className="text-[10px] text-slate-600 italic">待生成</span>
                                    )}
                                    
                                    <button 
                                      onClick={() => handleRegenerateSingleSentence(seq)}
                                      disabled={regeneratingSeq === seq}
                                      className="p-1 hover:bg-slate-800 rounded text-[10px] font-mono text-slate-400 hover:text-indigo-400 disabled:opacity-50 cursor-pointer"
                                      title="单独重新生成此句"
                                    >
                                      {regeneratingSeq === seq ? <Loader2 className="w-3 h-3 animate-spin" /> : "重合成"}
                                    </button>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ) : (
                        <div className="flex-1 flex items-center justify-center text-slate-500 text-xs">
                          请在左侧队列中选择一个批次以查看详细文本。
                        </div>
                      )}
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        )}

        {/* Tab 3: Stitch & Master Export Canvas */}
        {workspaceTab === 'master' && (
          <div className="flex-1 pt-20 pb-4 px-6 flex flex-col min-h-0 overflow-hidden">
            <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-0">
              
              {/* Left Column: Stitching controls */}
              <div className="space-y-4 overflow-y-auto pr-2 custom-scrollbar">
                <div className="glass-card p-6 bg-slate-900/40 text-center flex flex-col items-center justify-center relative overflow-hidden border-indigo-500/10">
                  <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 flex items-center justify-center mb-3">
                    <Music className="w-7 h-7 text-indigo-400" />
                  </div>
                  
                  <h3 className="text-base font-bold text-white mb-1">音频拼接与整书输出 (Layer 4)</h3>
                  <p className="text-xs text-slate-400 max-w-sm mb-4 leading-relaxed">
                    将所有已生成的单句音频（第 1 句至第 {globalSegments.length} 句）按照小说时间线进行高保真拼接（PCM 格式，24kHz 单声道 WAV）。
                  </p>

                  <div className="flex flex-col gap-2 w-full max-w-xs animate-in">
                    <button 
                      onClick={handleStitchMasterAudio}
                      disabled={globalSegments.length === 0 || pipelinePhase === 'stitching'}
                      className="btn-primary flex items-center justify-center gap-2 text-xs py-2.5 px-4 shadow-indigo-500/20 w-full cursor-pointer"
                    >
                      {pipelinePhase === 'stitching' ? (
                        <><Loader2 className="w-4 h-4 animate-spin" /> 正在拼接...</>
                      ) : (
                        <><Layers className="w-4 h-4" /> 合并拼接整书音频 WAV</>
                      )}
                    </button>
                    
                    <button 
                      onClick={handleExportFFmpegFiles}
                      disabled={globalSegments.length === 0}
                      className="btn-ghost flex items-center justify-center gap-2 text-xs py-2.5 px-4 w-full cursor-pointer"
                    >
                      <Download className="w-4 h-4" /> 导出 FFmpeg 脚本
                    </button>
                  </div>
                </div>

                {/* Master Audio Results */}
                {masterAudioUrl && (
                  <div className="glass-card p-5 border border-green-500/20 shadow-lg shadow-green-500/5 space-y-3">
                    <div className="flex items-center gap-3">
                      <CheckCircle2 className="w-4 h-4 text-green-400" />
                      <div>
                        <h4 className="font-bold text-xs text-white">有声书整书音频已就绪</h4>
                        <p className="text-[10px] text-slate-400">文件大小: {masterAudioBlob ? (masterAudioBlob.size / 1024 / 1024).toFixed(2) : 0} MB | 24kHz 单声道 WAV</p>
                      </div>
                    </div>
                    
                    <div className="flex flex-col gap-2 p-3 bg-slate-950/60 rounded-xl border border-slate-900">
                      <audio src={masterAudioUrl} controls className="w-full rounded h-8" />
                      <a 
                        href={masterAudioUrl} 
                        download={`${novelFileName || castData?.novel_name || 'audiobook'}_导出.wav`} 
                        className="btn-primary text-xs w-full flex items-center justify-center gap-2 py-2 cursor-pointer"
                      >
                        <Download className="w-4 h-4" /> 下载整书 WAV
                      </a>
                    </div>
                  </div>
                )}
              </div>

              {/* Right Column: Playable Sequential List (Paginated) */}
              <div className="glass-card p-5 flex flex-col min-h-0 bg-slate-950/20">
                <div className="flex justify-between items-center pb-3 border-b border-slate-800/40 mb-3 shrink-0">
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">小说全局音频播放列表 (逐句审计)</span>
                </div>
                
                <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar space-y-2">
                  {globalSegments.length === 0 ? (
                    <div className="text-center p-8 text-slate-500 text-xs mt-10">播放列表为空，请先运行 AI 剧本打标与 TTS 合成。</div>
                  ) : (() => {
                    const totalPages = Math.ceil(globalSegments.length / pageSize);
                    const paginatedSegments = globalSegments.slice((currentPage - 1) * pageSize, currentPage * pageSize);
                    
                    return (
                      <div className="space-y-2">
                        {paginatedSegments.map(seg => {
                          const speakerColor = getSpeakerColor(seg.char, castData?.cast || []);
                          const hasAudio = globalSequencedAudios.has(seg.seq!);
                          
                          return (
                            <div 
                              key={seg.seq} 
                              className={cn(
                                "p-2.5 rounded-lg border flex items-center justify-between gap-3 text-xs transition-colors",
                                playingSeq === seg.seq ? "bg-indigo-950/20 border-indigo-500/40" : "bg-slate-900/30 border-slate-850 hover:bg-slate-900/50"
                              )}
                            >
                              <div className="flex items-center gap-2 overflow-hidden">
                                <span className="font-mono text-[9px] text-slate-500 font-bold shrink-0">#{seg.seq}</span>
                                <span className="font-bold shrink-0 text-[11px]" style={{ color: speakerColor }}>{seg.char}</span>
                                <span className="text-slate-350 truncate">{seg.text}</span>
                              </div>
                              <div className="flex items-center gap-1.5 shrink-0">
                                {hasAudio ? (
                                  <button 
                                    onClick={() => handlePlayToggle(seg.seq!)}
                                    className="p-1 hover:bg-indigo-500/10 rounded text-indigo-400 cursor-pointer"
                                  >
                                    {playingSeq === seg.seq ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                                  </button>
                                ) : (
                                  <span className="text-[9px] text-slate-500 italic">无音频</span>
                                )}
                                <button 
                                  onClick={() => handleRegenerateSingleSentence(seg.seq!)}
                                  disabled={regeneratingSeq === seg.seq}
                                  className="p-1 hover:bg-slate-800 rounded text-[9px] font-mono text-slate-500 hover:text-indigo-400 cursor-pointer"
                                >
                                  {regeneratingSeq === seg.seq ? <Loader2 className="w-3 h-3 animate-spin" /> : "重合成"}
                                </button>
                              </div>
                            </div>
                          );
                        })}
                        
                        {/* Pagination Controls */}
                        {totalPages > 1 && (
                          <div className="flex items-center justify-between pt-3 border-t border-slate-800/40 mt-3 shrink-0">
                            <button
                              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                              disabled={currentPage === 1}
                              className="px-2.5 py-1.5 rounded-lg bg-slate-850 hover:bg-slate-800 text-xs font-semibold text-slate-300 disabled:opacity-30 disabled:pointer-events-none transition-colors cursor-pointer"
                            >
                              上一页
                            </button>
                            <span className="text-[11px] text-slate-400 font-mono">
                              页码 {currentPage} / {totalPages} (共 {globalSegments.length} 句)
                            </span>
                            <button
                              onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                              disabled={currentPage === totalPages}
                              className="px-2.5 py-1.5 rounded-lg bg-slate-850 hover:bg-slate-800 text-xs font-semibold text-slate-300 disabled:opacity-30 disabled:pointer-events-none transition-colors cursor-pointer"
                            >
                              下一页
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>
              </div>

            </div>
          </div>
        )}

        {/* 📟 底端终端显示框 (Terminal Console) */}
        <div className="h-48 border-t border-slate-800/80 bg-[#05050a] flex flex-col shrink-0 z-20">
          <div className="flex items-center justify-between px-4 py-1.5 bg-slate-900/50 border-b border-slate-800/50">
            <div className="flex items-center gap-2">
              <Terminal className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-[10px] font-bold text-slate-300 uppercase tracking-widest">系统终端日志 (Console)</span>
            </div>
            <button onClick={() => setLogs([])} className="text-[9px] text-slate-500 hover:text-slate-300 cursor-pointer">清空</button>
          </div>
          <div className="flex-1 p-3 overflow-y-auto custom-scrollbar font-mono text-[10px] space-y-1.5 flex flex-col-reverse">
            <div className="space-y-1 w-full pb-2">
              {logs.map((log, idx) => (
                <div key={idx} className={cn(
                  "grid grid-cols-[65px_1fr] gap-2 hover:bg-white/5 px-1 rounded transition-colors",
                  log.type === 'error' ? "text-red-400" : 
                  log.type === 'success' ? "text-emerald-400" : 
                  log.type === 'warn' ? "text-amber-400" : "text-slate-450"
                )}>
                  <span className="text-slate-600 opacity-60">[{log.time}]</span>
                  <span className="break-words font-mono font-medium leading-relaxed">{log.msg}</span>
                </div>
              ))}
              {logs.length === 0 && <div className="text-slate-700 text-center mt-2">系统初始化完成，等待任务指令...</div>}
            </div>
          </div>
        </div>
      </div>      {/* 🟢 END OF MIDDLE PANEL */}

      {/* 🟢 右侧：系统配置与控制台控制 (Setup Panel & Event Logs) */}
      {!isRightPanelCollapsed && (
        <div className="w-[340px] border-l border-slate-800/50 bg-[#0a0a14] flex flex-col z-10 shrink-0">
        
        {/* Tab Header Selector */}
        <div className="flex border-b border-slate-800/50 glass-panel">
          <button 
            onClick={() => setActiveTab('setup')}
            className={cn("flex-1 py-3.5 text-[10px] font-bold uppercase tracking-wider transition-colors", activeTab === 'setup' ? "text-indigo-400 border-b-2 border-indigo-400 bg-indigo-500/5" : "text-slate-500 hover:text-slate-300")}
          >
            Setup
          </button>

          <button 
            onClick={() => setActiveTab('cast')}
            className={cn("flex-1 py-3.5 text-[10px] font-bold uppercase tracking-wider transition-colors", activeTab === 'cast' ? "text-indigo-400 border-b-2 border-indigo-400 bg-indigo-500/5" : "text-slate-500 hover:text-slate-300")}
          >
            Cast Registry
          </button>
        </div>

        {/* Setup Configuration Content */}
        {activeTab === 'setup' && (
          <div className="flex-1 overflow-y-auto p-5 space-y-5 custom-scrollbar">
            
            {/* API Keys */}
            <div>
              <label className="flex items-center gap-2 text-xs font-bold text-slate-400 mb-2 uppercase tracking-wide">
                <Settings className="w-3.5 h-3.5" /> API Keys (New line per key)
              </label>
              <textarea 
                className="input-dark font-mono text-[10px] h-20 leading-relaxed resize-none" 
                placeholder="AIzaSy..."
                value={apiKeysStr} onChange={e => setApiKeysStr(e.target.value)} 
              />
            </div>
            
            {/* Cast Config (JSON) */}
            <div>
              <label className="flex items-center gap-2 text-xs font-bold text-slate-400 mb-2 uppercase tracking-wide">
                <Users className="w-3.5 h-3.5" /> Cast Config (JSON)
              </label>
              <textarea 
                className="input-dark font-mono text-[10px] h-28 leading-relaxed resize-none" 
                placeholder='{"novel_name": "...", "cast": []}'
                value={castJsonStr} onChange={e => setCastJsonStr(e.target.value)} 
              />
            </div>

            {/* Chunk Size Configuration */}
            <div>
              <label className="flex items-center gap-2 text-xs font-bold text-slate-400 mb-2 uppercase tracking-wide">
                <Settings className="w-3.5 h-3.5" /> Chunk Size (Chars)
              </label>
              <input 
                type="number" 
                className="input-dark font-mono text-[10px] w-full"
                value={chunkSize} onChange={e => setChunkSize(Number(e.target.value))} 
              />
            </div>
            
            {/* Manuscript Upload & Paste */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="flex items-center gap-2 text-xs font-bold text-slate-400 uppercase tracking-wide">
                  <FileJson className="w-3.5 h-3.5" /> Novel Manuscript
                  {novelFileName && (
                    <span className="text-[9px] text-indigo-300 bg-indigo-500/10 px-1.5 py-0.5 rounded border border-indigo-500/20 truncate max-w-[120px] ml-1">
                      {novelFileName}.txt
                    </span>
                  )}
                </label>
                <div className="relative overflow-hidden">
                  <button type="button" className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] font-bold border border-slate-700 transition-all cursor-pointer shadow">
                    <Upload className="w-3 h-3 text-indigo-400" /> Upload TXT
                  </button>
                  <input 
                    type="file" 
                    accept=".txt" 
                    onChange={handleFileUpload} 
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" 
                  />
                </div>
              </div>
              <textarea 
                className="input-dark text-[10px] h-28 leading-relaxed resize-none font-serif custom-scrollbar" 
                placeholder="Paste content here or upload a TXT file..."
                value={rawText} onChange={e => setRawText(e.target.value)} 
              />
            </div>
            
            {/* 高级提示词配置 (折叠) */}
            <details className="group border border-slate-800/50 rounded-xl overflow-hidden">
              <summary className="flex items-center gap-2 text-xs font-bold text-slate-500 uppercase tracking-wide cursor-pointer hover:text-slate-300 transition-colors p-3 select-none bg-slate-900/30">
                <Settings className="w-3.5 h-3.5" />
                <span>高级提示词模板 (通常无需修改)</span>
                <ChevronRight className="w-3 h-3 transition-transform group-open:rotate-90 ml-auto" />
              </summary>
              <div className="space-y-4 p-4 border-t border-slate-800/40">
                <div>
                  <label className="flex items-center gap-1.5 text-[10px] font-bold text-slate-400 mb-1.5 uppercase tracking-wide">
                    L2 打标提示词
                  </label>
                  <p className="text-[9px] text-slate-500 mb-1.5 leading-relaxed">
                    剧本分段与角色标注的系统指令。占位符: <code className="text-indigo-300 font-mono text-[9px] bg-slate-900 px-1 py-0.5 rounded">{"{{cast_json}}"}</code> <code className="text-indigo-300 font-mono text-[9px] bg-slate-900 px-1 py-0.5 rounded">{"{{previous_text}}"}</code> <code className="text-indigo-300 font-mono text-[9px] bg-slate-900 px-1 py-0.5 rounded">{"{{current_chunk}}"}</code>
                  </p>
                  <textarea 
                    className="input-dark font-mono text-[9px] h-40 leading-relaxed resize-y custom-scrollbar"
                    value={taggingSystemPrompt}
                    onChange={e => setTaggingSystemPrompt(e.target.value)}
                  />
                </div>
              </div>
            </details>

            <button 
              onClick={handleParseAndChunk} 
              disabled={!rawText || !castJsonStr}
              className="w-full bg-slate-800 hover:bg-slate-700 text-slate-200 py-3 rounded-xl shadow-lg border border-slate-705 transition flex justify-center items-center gap-2 font-bold text-xs cursor-pointer"
            >
              <Upload className="w-4 h-4 text-indigo-400" /> Layer 1: Smart Chunking
            </button>
            <button 
              onClick={handleClearProject}
              className="w-full bg-red-950/20 hover:bg-red-950/40 text-red-400 py-3 rounded-xl shadow-lg border border-red-900/30 transition flex justify-center items-center gap-2 font-bold text-xs cursor-pointer"
            >
              <AlertCircle className="w-4 h-4 text-red-400" /> 清空当前工程 (重置)
            </button>
          </div>
        )}



        {/* Cast Configuration Content */}
        {activeTab === 'cast' && (
          <div className="flex-1 overflow-y-auto p-5 space-y-3 custom-scrollbar">
            {castData ? (
              <>
                <div className="glass-card p-3 mb-3 bg-slate-800/10 border-slate-800/60">
                  <div className="text-[9px] text-slate-500 uppercase tracking-widest mb-1">Active Project</div>
                  <div className="font-bold text-xs text-indigo-300">{castData.novel_name}</div>
                </div>
                
                {castData.cast.map(c => (
                  <div 
                    key={c.character_name} 
                    className="text-xs p-3 rounded-xl border border-slate-850 glass-card flex justify-between items-center group cursor-pointer hover:border-slate-800 transition" 
                    style={{ borderLeft: `3px solid ${c.colorHex}` }}
                  >
                    <div className="flex flex-col gap-0.5 overflow-hidden">
                       <span className="font-bold text-slate-200">{c.character_name}</span>
                       <span className="text-[9px] text-slate-500 truncate max-w-[150px]">别名: {c.aliases.join(', ')}</span>
                    </div>
                    <div className="flex flex-col items-end gap-1">
                      <span className="font-mono text-[9px] text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/15">{c.assigned_voice_id}</span>
                    </div>
                  </div>
                ))}
              </>
            ) : (
              <div className="text-center text-slate-500 mt-10 text-xs">Cast registry is empty. Configure in Setup Tab.</div>
            )}
          </div>
        )}
      </div>
      )}

      {/* Project Management Modal */}
      {isProjectModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-[#0f111a] border border-slate-800 rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl flex flex-col max-h-[80vh]">
            <div className="p-5 border-b border-slate-800/80 flex items-center justify-between bg-slate-900/30">
              <div className="flex items-center gap-3">
                <FolderOpen className="w-5 h-5 text-indigo-400" />
                <h2 className="text-lg font-bold text-slate-200 tracking-tight">历史任务管理 / 工作区</h2>
              </div>
              <button 
                onClick={() => setIsProjectModalOpen(false)}
                className="text-slate-500 hover:text-slate-300 transition-colors p-1 text-xl leading-none"
              >
                &times;
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-5 space-y-3 custom-scrollbar">
              {projectList.length === 0 ? (
                <div className="text-center py-10 text-slate-500 text-sm">
                  暂无历史任务
                </div>
              ) : (
                projectList.sort((a, b) => b.updatedAt - a.updatedAt).map(p => (
                  <div 
                    key={p.id} 
                    className={cn(
                      "flex items-center justify-between p-4 rounded-xl border transition-all",
                      p.id === projectId 
                        ? "bg-indigo-500/10 border-indigo-500/30 shadow-lg shadow-indigo-500/5" 
                        : "glass-card border-slate-800 hover:border-slate-700 hover:bg-slate-800/40"
                    )}
                  >
                    <div className="flex flex-col gap-1">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-slate-200 text-sm">{p.name}</span>
                        {p.id === projectId && (
                          <span className="text-[9px] uppercase tracking-widest px-1.5 py-0.5 rounded bg-indigo-500 text-white font-bold">
                            当前工作区
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-slate-500 flex items-center gap-4">
                        <span>最后修改: {new Date(p.updatedAt).toLocaleString()}</span>
                        <span>分块数: {p.totalChunks || 0}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {p.id !== projectId && (
                        <button 
                          onClick={() => loadProject(p.id)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 hover:bg-indigo-500/20 transition-colors text-xs font-semibold"
                        >
                          <FolderOpen className="w-3.5 h-3.5" />
                          载入
                        </button>
                      )}
                      <button 
                        onClick={() => deleteProject(p.id)}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 transition-colors text-xs font-semibold"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                        删除
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
            
            <div className="p-4 border-t border-slate-800/80 bg-slate-900/30 flex justify-end">
              <button 
                onClick={createNewProject}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-colors text-sm font-bold shadow-lg shadow-indigo-500/20"
              >
                <Plus className="w-4 h-4" />
                新建空白任务
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}