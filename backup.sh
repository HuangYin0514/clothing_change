####################################
# git
####################################
VERSION_NAME="init"


git status # 1. 查看状态
git add . # 2. 添加所有修改
git commit -m "feat: ${VERSION_NAME}" # 3. 提交到本地（替换提交说明）
git pull origin main # 4. 拉取远程最新代码
git push origin main # 5. 推送到远程\
echo -e "✅ 成功：代码已提交并同步到远程分支！"

echo -e "🎉 所有操作执行完成！"


####################################
# 拷贝
####################################
# rm -rf main
# cp -rf _历史代码/version000 main
# cp -rf main "_历史代码/${VERSION_NAME}"
# echo "\n✅ 成功：文件已拷贝！"