# _*_ coding: utf-8 _*_

import os
from PIL import Image, ImageFont, ImageDraw, ImageChops

def takeUnicode(elem):
    return elem.encode('unicode_escape').decode()

def getFontImage(text,size,style,font,y_offset):
    im=Image.new("RGB",(size,size),(255,255,255))
    dr=ImageDraw.Draw(im)

    path = os.path.join("../inputFiles/",style,font)
    font=ImageFont.truetype(path,size)
    dr.text((0,y_offset),text,font=font,fill="#000000")
    im = im.convert('1')
    return im

def isEmpty(im):
    return not ImageChops.invert(im).getbbox()

def main():
    font_dir = "../inputFiles/font/"
    font_list = os.listdir(font_dir)

    char = '智'
    unicode = takeUnicode(char)[2:]
    style = "font"
    size = 256

    write_directory = os.path.dirname('../characterImages/{}/'.format(style))
    if not os.path.exists(write_directory):
        os.makedirs(write_directory)

    good_fonts = []

    count = 0

    for font in font_list:
        try:
            im = getFontImage(char,size,style,font,0)
            if isEmpty(im):
                print(font)
                pass
            else:
                good_fonts.append(count)
                im.save('../characterImages/{}/{}.png'.format(style,font[:-4]))
        except Exception as e:
            print(e)
            pass
        count += 1
    print("Len:")
    print(len(good_fonts))
    print("Good fonts:")
    for num in good_fonts:
        print(num)

def main2():
    # characters = "一丁丂七丅丆万丈三上下丌不与丐丑专且丕世丘丙业丛东丝丞丢两严丧丨丩个丫丬中丰串临丵丶丷丹主丽举丿乀乂乃久乇么之乌乍乎乏乐乒乓乔乖乘乙乚乛乜九乞习乡书乩买乱乳乾亅了予争事二亍于亏云互亓五井亘亚些亟亠亡亢交亥亦产亨亩享京亭亮亲亳亵人亻亼亾亿什仁仂仃仄仅仆仇仉今介仌仍从仑仓仔仕他仗付仙仝仞仟仡代令以仨仪仫们仰仲仳仵件价任份仿企伉伊伍伎伏伐休众优伙会伛伞伟传伢伤伥伦伧伪伫伯估伲伴伶伸伺似伽佃但位低住佐佑体何佗佘余佚佛作佝佞佟你佣佤佥佧佩佬佯佰佳佴佶佻佼佾使侃侄侈侉例侍侏侑侔侗供依侠侣侥侦侧侨侩侪侬侮侯侵便促俄俅俊俎俏俐俑俗俘俚俜保俞俟信俣俦俨俩俪俭修俯俱俳俸俺俾倌倍倏倒倔倘候倚倜借倡倥倦倨倩倪倬倭倮债值倾偃假偈偌偎偏偕做停健偬偶偷偻偾偿傀傅傈傍傣傥傧储傩催傲傺傻像僖僚僦僧僬僭僮僳僵僻儆儇儋儒儡儿兀允元兄充先光克免兑兒兕兖党兜兢入全八公六兮兰共关兴兵其具典兹养兼兽冀冁冂冃内冈冉冊冋册再冎冏冒冕冖冗写军农冝冠冢冤冥冫冬冯冰冱冲决况冶冷冻冼冽净凄准凇凉凋凌减凑凛凝几凡凤凫凭凯凰凳凵凶凸凹出击凼函凿刀刁刂分切刈刊刍刎刑划刖列刘则刚创初删判刨利别刭刮到刳制刷券刹刺刻刽刿剀剁剂剃削剌前剐剑剔剖剜剞剡剥剧剩剪副割剽剿劁劂劈劐劓力劝功加务劢劣劦动助努劫劬劭励劲劳劾势勃勇勉勋勐勒勖勘募勤勰勹勺勾勿匀匃包匈匋匍匏匐匕化北匙匚匝匠匡匣匦匪匮匹区医匽匾匿十千卅午卉半华协卑卒卓单卖南博卜卞卟占卡卢卣卤卦卧卩卫卬卮卯危即却卷卸卺卿厂厃厄厅历厉压厌厍厕厘厚厝原厢厣厥厦厨厩厮厶去县叁参又叉及友双反发叒叔叕取受变叙叚叛叟叠口古句另叨叩只叫召叭叮可台叱史右叵叶号司叹叻叼叽吁吃各吆合吉吊同名后吐向吒吓吕吖吗君吝吞吟吠吡吣否吧吨吩含听吭吮启吱吲吴吵吸吹吻吼吾呀呃呆呈告呋呐呒呓呔呕呖呗员呙呛呜呢呤呦周呱呲味呵呶呷呸呻呼命咀咂咄咅咆咋和咎咏咐咒咔咕咖咙咚咛咝咣咤咦咧咨咩咪咫咬咭咯咱咳咴咸咻咽咿哀品哂哄哆哇哈哉哌响哎哏哐哑哒哓哔哕哗哙哚哜哝哞哟哥哦哧哨哩哪哭哮哲哳哺哼哽哿唁唆唇唉唏唐唑唔唛唠唢唣唤唧唪唬售唯唰唱唳唷唼唾唿啁啃啄商啉啊啐啕啖啜啡啤啥啦啧啪啬啭啮啵啶啷啸啻啼啾喀喁喂喃善喇喈喉喊喋喏喑喔喘喙喜喝喟喧喱喳喵喷喹喻喽喾喿嗄嗅嗉嗌嗍嗑嗒嗓嗔嗖嗜嗝嗟嗡嗣嗤嗥嗦嗨嗪嗫嗬嗯嗲嗳嗵嗷嗽嗾嘀嘁嘈嘉嘌嘎嘏嘘嘛嘞嘟嘣嘤嘧嘬嘭嘱嘲嘴嘶嘹嘻嘿噌噍噎噔噗噘噙噜噢噤器噩噪噫噬噱噶噻噼嚅嚆嚎嚏嚓嚣嚯嚷嚼囊囔囗囚四囝回囟因囡团囤囫囬园困囱围囵囹固国图囿圃圄圆圈圉圊圜土圡圣在圩圪圬圭圮圯地圳圹场圻圾址坂均坊坌坍坎坏坐坑块坚坛坜坝坞坟坠坡坤坦坨坩坪坫坭坯坳坶坷坻坼垂垃垄垅堇塞士壬壴壹夂处夊夋夌夏夕夗多大天太夫夭央失夲头夷夸夹奂奄奇奉奎奏奥女奴妟妻妾姆委娄婴子孔字孚孛孝孟宀宁它宅官定宛客宣害宾密察寸寺寿尃小少尔尗尚尝尞尢尧就尸尹尺尼居屈屋展属屮屯屰山岁岂崔崩巛川巢工左巫差己巾币布希帝干幸幺幼广府廴建廾廿开弋弓弔引弗弚弟彐彑录彖彡彭彳心忄忽思怱恩意我戚戛户戾扁手扌才执折拉拍支攴攵故敖敬文斗斤斯方旁族无既日旦昊昌易昔昗昚是曰曲更曷曹曾最朁月有朋朔朝木朩未本朱朵朿杀束松林果枼查栗桑欠次欮欶此歹殳每比毕毛氏氐气水氵氶永氺求波泰海火灬灰炎焦爪爫爱爵父爹爻爿片牙牛牟犭玄玉王瓦甘甚生用甫甬田由甲申电甹畏畐畺畾疋疐疑疒癶登白百皂皆皇皋皮皿益盍目直相真睘矛矢矣石示礻票祭禀禁禸禹禺禽禾秋秦穴空立童竹筮米粟糸素索纟约缶罒羊美羽翁老耂者耆而耑耒耳聂聿肀肉肖肯育胃臣自臭至臼舌舍舟艮色艹艺艾苗苟若荅荷菐葛蒦蒿虍虎虚虫蚩血行衣衤襄西覀见角觜言詹讠诸谷豆豕象豦豪豸贝贞责贲贵赤走足身车轨转辛辟辰辶达通邑那郎都采里金钅镸长门阜阝队阿隶隹隻难雨需霍青非面革韦韭音页顷风飞饣首香马骨高髟鬯鬲鬼鱼鲁鹿麦麻黄黍黑黹鼎鼓鼠鼻齐齿龠"
    # char_sorted = list(characters)
    # char_sorted.sort(key=takeUnicode)

    characters = []
    for i in range(0x4E00, 0x9FA5+1): 
        characters.append(chr(i))

    size = 1000
    fonts = ["HYQiHei-25JF.otf", "HanYiXiXingKaiJian-1.ttf", "Kaiti_2.TTF", "SourceHanSansCN-Regular.otf", "HYXinRenWenSongW-1.otf","SourceHanSerifCN-Heavy-4.otf"]
    styles = ["Heiti", "Kaiti", "Kaiti", "Regular", "Songti", "Songti"]
    adds = ["_2", "_1", "_2", "", "_1", "_2"]
    offsets = [-70, 0, -20, 0, -60, -260]
    assert(len(fonts)==len(styles))
    assert(len(styles)==len(adds))
    assert(len(adds)==len(offsets))

    for i in range(len(fonts)):
        style = styles[i]
        add = adds[i]
        write_directory = os.path.dirname('../characterImages/{}/'.format(style+add))
        if not os.path.exists(write_directory):
            os.makedirs(write_directory)

    for i in range(len(fonts)):
        if (i != 2):
            continue
        font = fonts[i]
        style = styles[i]
        add = adds[i]
        y_offset = offsets[i]
        for char in characters:
            unicode = takeUnicode(char)[2:] 
            try:
                im = getFontImage(char,size,style,font,y_offset)
                if isEmpty(im):
                    pass
                    # print("Character missing for font: " + style + "/" + font)
                    # print("Unicode: ", unicode)
                else:
                    im.save('../characterImages/{}/{}.png'.format(style+add,unicode))
            except Exception as error:
                pass
                # print("Exception: ", error)
                # print("For unicode: ", unicode)

if __name__ == '__main__':
    main()
