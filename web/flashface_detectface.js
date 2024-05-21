import { app } from '../../scripts/app.js'
console.log("FlashFace Node for Comfy");
app.registerExtension({
    name: "flashface.detectface",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "FlashFaceDetectFace") {
            // add dynamic input for image1, imgae2, etc.
            const onNodeCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
                this.addInput(`image1`, 'PIL_IMAGE')
                return r
            }
            // add onConnectionsChange for dynamic input
            const onConnectionsChange = nodeType.prototype.onConnectionsChange
            nodeType.prototype.onConnectionsChange = function (
                type,
                index,
                connected,
                link_info,
            ) {
                const r = onConnectionsChange
                    ? onConnectionsChange.apply(this, arguments)
                    : undefined
                dynamic_connection(this, index, connected, 'image', 'PIL_IMAGE')
            }

            const dynamic_connection = (
                node,
                index,
                connected,
                connectionPrefix = 'image',
                connectionType = 'PSDLAYER',
                nameArray = [],
            ) => {
                if (!node.inputs[index].name.startsWith(connectionPrefix)) {
                    return
                }

                // remove all non connected inputs
                if (!connected && node.inputs.length > 1) {
                    console.log(`Removing input ${index} (${node.inputs[index].name})`)
                    if (node.widgets) {
                        const w = node.widgets.find((w) => w.name === node.inputs[index].name)
                        if (w) {
                            w.onRemoved?.()
                            node.widgets.length = node.widgets.length - 1
                        }
                    }
                    node.removeInput(index)

                    // make inputs sequential again
                    for (let i = 0; i < node.inputs.length; i++) {
                        const name = i < nameArray.length ? nameArray[i] : `${connectionPrefix}${i + 1}`
                        node.inputs[i].label = name
                        node.inputs[i].name = name
                    }
                }

                // add an extra input
                if (node.inputs[node.inputs.length - 1].link != undefined) {
                    const nextIndex = node.inputs.length
                    const name = nextIndex < nameArray.length ? nameArray[nextIndex] : `${connectionPrefix}${nextIndex + 1}`

                    console.log(`Adding input ${nextIndex + 1} (${name})`)

                    node.addInput(name, connectionType)
                }
            }

        }

    },
});

